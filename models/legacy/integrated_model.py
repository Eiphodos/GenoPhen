import torch
import torch.nn as nn


def ast_accuracy_2(predicted_values, real_values):        # also do based on R and S to get very major error rate!
    correct_count = 0
    for i in range(len(real_values)):
        if torch.argmax(predicted_values[i]) == real_values[i]:
            correct_count += 1
    if len(real_values) == 0:
        return 0
    return correct_count/len(real_values)


class integrated_model():

    def __init__(self, cfg, ast_model, amr_model, train_dataloader, eval_dataloader, num_epochs, weights_train, weights_val,
                 weights_r_val, weights_s_val, df_details, hidden_dim):
        self.num_epochs = num_epochs
        self.weights_train = weights_train
        self.weights_val = weights_val
        self.df_details = df_details
        self.weights_s_val = weights_s_val
        self.weights_r_val = weights_r_val
        self.ast_model = ast_model
        self.amr_model = amr_model
        self.device = cfg['device']
        self.hidden_dim = hidden_dim
        self.number_ab = len(cfg['antibiotics']['antibiotics_in_use'])
        self.abpred = [self.ab_pred(hidden_dim) for _ in range(self.number_ab)]
        self.losses = [nn.CrossEntropyLoss(weight=torch.tensor(v, requires_grad=False).to(self.device),
                                           reduction='mean') for v in cfg['antibiotics_weights'].values()]

        self.acc = [ast_accuracy_2 for _ in range(self.number_ab)]
        self.train_dataloader = train_dataloader

        long_param_list = []
        for i in range(len(self.abpred)):
            long_param_list.extend(list(self.abpred[i].parameters()))

        self.optimizer = torch.optim.Adam(list(self.ast_model.net.parameters()) +
                                          list(self.amr_model.parameters()) + long_param_list,
                                          lr=config['parameters']['optimizer']['lr'] * 0.5,
                                          weight_decay=config['parameters']['optimizer']['weight_decay'])
        self.epoch = 0
        self.train_counter = 0
        self.val_counter = 0
        self.summarywrapper = None
        self.val_loader = eval_dataloader
        self.start_ab_stats = False

        self.n_batches_train = np.ceil(len(self.train_dataloader.dataset) / config_2.batch_size)
        self.n_batches_val = np.ceil(len(self.val_loader.dataset) / config_2.batch_size)
        self.eval_every_n = 2
        self.num_training_steps = self.num_epochs * len(self.train_dataloader)
        self.scheduler = get_scheduler("linear", optimizer=self.optimizer, num_warmup_steps=0,
                                       num_training_steps=self.num_training_steps)

    def ab_pred(self, hidden_dim):
        self.ab = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                nn.ReLU(), nn.LayerNorm(hidden_dim),
                                nn.Linear(hidden_dim, 2)).to(self.device)
        return self.ab

    def forward(self, input_ids, x, positions_y, batch_index_y, positions_x, batch_index_x, x_valid_len_to,
                attention_mask):
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        outputs = self.amr_model(input_ids, attention_mask)  # index out of range of self
        output_logits = outputs.logits
        cls_tokens_amr = output_logits[:, 0, :]

        cls_tokens_ab = self.ast_model.net.encoder(x, x_valid_len_to)[:, 0, :]
        encoded_x = torch.cat([cls_tokens_amr, cls_tokens_ab], dim=1).to(self.device)

        encoded_x = encoded_x[:, None, :]
        mlm_hat = torch.cat([a(encoded_x) for a in self.abpred], 1).to(self.device)

        ab_y_hat = mlm_hat[batch_index_y, positions_y]
        ab_x_hat = mlm_hat[batch_index_x, positions_x]

        return ab_y_hat, ab_x_hat

    def predict(self, input_ids, x, x_pos_antibiotic, y_pos_antibiotic, x_resp, len_x, len_y, total_len_x,
                attention_mask, deterministic=False):
        if deterministic:
            self.net.eval()
        # Collect for y prediction
        x = x.to(self.device)
        total_len_x = total_len_x.to(self.device)
        tmp = [[ly] * len_y[ly] for ly in range(len(len_y))]
        batch_index_y = torch.tensor(list(itertools.chain(*tmp))).to(self.device)
        tmp = torch.reshape(y_pos_antibiotic, (-1,))
        positions_y = tmp[tmp >= 0].to(self.device)

        # Collect for x prediction
        tmp = [[lx] * len_x[lx] for lx in range(len(len_x))]
        batch_index_x = torch.tensor(list(itertools.chain(*tmp))).to(self.device)
        tmp = torch.reshape(x_pos_antibiotic, (-1,))
        positions_x = tmp[tmp >= 0].to(self.device)
        tmp = torch.reshape(x_resp, (-1,))
        ab_x_true = tmp[tmp >= 0].to(self.device)

        ab_y_hat, ab_x_hat = self.forward(input_ids, x, positions_y, batch_index_y,
                                          positions_x, batch_index_x, total_len_x, attention_mask)

        return ab_y_hat, ab_x_hat, ab_x_true, positions_y, positions_x, batch_index_y, batch_index_x

    def forward_pass_loss(self, input_ids, x, x_pos_antibiotic, y_pos_antibiotic, x_resp, y_resp, len_x, len_y,
                          total_len_x, attention_mask,
                          deterministic=False):
        tmp = torch.reshape(y_resp, (-1,))
        ab_y_true = tmp[tmp >= 0].to(self.device)
        ab_y_hat, ab_x_hat, ab_x_true, positions_y, positions_x, batch_index_y, batch_index_x = \
            self.predict(input_ids, x, x_pos_antibiotic, y_pos_antibiotic, x_resp, len_x, len_y, total_len_x,
                         attention_mask, deterministic)

        return ab_y_hat, ab_x_hat, ab_y_true, ab_x_true, positions_y, positions_x, batch_index_y, batch_index_x

    def train_pass_and_loss(self, input_ids, x, x_pos_antibiotic, y_pos_antibiotic, x_resp, y_resp, len_x, len_y,
                            total_len_x, attention_mask, val=False):

        ab_y_hat, ab_x_hat, ab_y_true, ab_x_true, positions_y, positions_x, _, _ = \
            self.forward_pass_loss(input_ids, x, x_pos_antibiotic, y_pos_antibiotic, x_resp, y_resp, len_x, len_y,
                                   total_len_x, attention_mask)

        index_list = []
        for i in range(len(positions_y)):
            if len(ab_y_hat[positions_y == i]) > 0:
                index_list.append(i)

        loss_y = [self.losses[i](ab_y_hat[positions_y == i, :], ab_y_true[positions_y == i])
                  for i in range(self.number_ab) if len(ab_y_true[positions_y == i]) > 0]

        loss_x = [self.losses[i](ab_x_hat[positions_x == i, :], ab_x_true[positions_x == i])
                  for i in range(self.number_ab)]

        acc = [self.acc[i](ab_y_hat[positions_y == i, :], ab_y_true[positions_y == i])
               for i in range(self.number_ab)]

        error_rates = [error_rates_2(ab_y_hat[positions_y == i, :], ab_y_true[positions_y == i])
                       for i in range(self.number_ab)]

        me = [error_rates[i][0] for i in range(len(error_rates))]
        vme = [error_rates[i][1] for i in range(len(error_rates))]

        loss_y = sum(loss_y) / len(loss_y)
        loss_x = sum(loss_x) / len(loss_x)
        acc = sum(acc) / len(index_list)
        me = sum(me) / len(index_list)
        vme = sum(vme) / len(index_list)
        return ab_y_hat, ab_x_hat, ab_y_true, ab_x_true, loss_y, loss_x, acc, me, vme

    def val_pass_and_loss(self, input_ids, x, x_pos_antibiotic, y_pos_antibiotic, x_resp, y_resp, len_x, len_y,
                          total_len_x, attention_mask, val=False):
        with torch.no_grad():
            ab_y_hat, ab_x_hat, ab_y_true, ab_x_true, positions_y, positions_x, _, _ = \
                self.forward_pass_loss(input_ids, x, x_pos_antibiotic, y_pos_antibiotic, x_resp, y_resp, len_x, len_y,
                                       total_len_x, attention_mask)

            loss_y = [self.losses[i](ab_y_hat[positions_y == i, :], ab_y_true[positions_y == i])
                      for i in range(self.number_ab) if len(ab_y_true[positions_y == i]) > 0]

            loss_x = [self.losses[i](ab_x_hat[positions_x == i, :], ab_x_true[positions_x == i])
                      for i in range(self.number_ab) if len(ab_x_hat[positions_x == i]) > 0]

            acc_list = [self.acc[i](ab_y_hat[positions_y == i, :], ab_y_true[positions_y == i])
                        for i in range(self.number_ab)]

            error_rates = [error_rates_2(ab_y_hat[positions_y == i, :], ab_y_true[positions_y == i])
                           for i in range(self.number_ab)]

            me_list = [error_rates[i][0] for i in range(len(error_rates))]
            vme_list = [error_rates[i][1] for i in range(len(error_rates))]

            if self.start_ab_stats == True and val == True:
                ab_spec_stat(index_list, s_index, r_index, acc_list, me_list, vme_list)

            loss_y = sum(loss_y) / len(loss_y)
            loss_x = sum(loss_x) / len(loss_x)
            # acc = sum(acc_list)/len(index_list)
            # me = sum(me_list)/len(s_index)
            # vme = sum(vme_list)/len(r_index)
            return ab_y_hat, ab_x_hat, ab_y_true, ab_x_true, loss_y, loss_x, positions_y

    def _train_epoch(self):
        # Set Train Mode
        # Hur fick vi batch från början?
        self.amr_model.train()

        train_loss = 0.
        acc_tot = 0
        me_tot = 0
        vme_tot = 0
        # for input_ids, x, x_pos_antibiotic, y_pos_antibiotic, x_resp, y_resp, len_x, len_y, total_len_x, attention, labels in self.train_loader:
        for batch in tqdm(self.train_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            input_ids = batch['input_ids']
            x = batch['ab']
            x_pos_antibiotic = batch['x pos ab']
            y_pos_antibiotic = batch['y pos ab']
            x_resp = batch['x resp']
            y_resp = batch['y resp']
            len_x = batch['len x']
            len_y = batch['len y']
            total_len_x = batch["total len x"]
            attention = batch['attention_mask']
            self.optimizer.zero_grad()
            ab_y_hat, ab_x_hat, ab_y_true, ab_x_true, loss_y, loss_x, acc, me, vme = \
                self.train_pass_and_loss(input_ids, x, x_pos_antibiotic, y_pos_antibiotic, x_resp, y_resp, len_x, len_y,
                                         total_len_x, attention)

            loss = loss_y  # + loss_x

            # Backward Pass
            loss.backward()
            train_loss += loss.item()

            acc_tot += acc
            me_tot += me
            vme_tot += vme

            n_batch = self.n_batches_train
            self.train_counter += 1
            # Update the Weights
            self.optimizer.step()
        self.epoch += 1
        if self.scheduler:
            self.scheduler.step()
        return train_loss / self.n_batches_train, acc_tot / self.n_batches_train, me_tot / self.n_batches_train, vme_tot / self.n_batches_train

    def _val_epoch(self):
        # Set Train Mode
        # Hur fick vi batch från början?
        self.amr_model.eval()

        train_loss = 0.
        acc_tot = 0
        me_tot = 0
        vme_tot = 0

        ab_y_hat_tot = torch.empty(0).to(self.device)
        ab_y_true_tot = torch.empty(0).to(self.device)
        positions_y_tot = torch.empty(0).to(self.device)
        # for input_ids, x, x_pos_antibiotic, y_pos_antibiotic, x_resp, y_resp, len_x, len_y, total_len_x, attention, labels in self.train_loader:
        for batch in tqdm(self.val_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            input_ids = batch['input_ids']
            x = batch['ab']
            x_pos_antibiotic = batch['x pos ab']
            y_pos_antibiotic = batch['y pos ab']
            x_resp = batch['x resp']
            y_resp = batch['y resp']
            len_x = batch['len x']
            len_y = batch['len y']
            total_len_x = batch["total len x"]
            attention = batch['attention_mask']
            # self.optimizer.zero_grad()
            ab_y_hat, ab_x_hat, ab_y_true, ab_x_true, loss_y, loss_x, positions_y = \
                self.val_pass_and_loss(input_ids, x, x_pos_antibiotic, y_pos_antibiotic, x_resp, y_resp, len_x, len_y,
                                       total_len_x, attention, val=True)
            # loss = loss_y  # + loss_x

            # Backward Pass
            # loss.backward()
            # train_loss += loss.item()

            ab_y_hat_tot = torch.cat([ab_y_hat_tot, ab_y_hat], dim=0).to(self.device)
            ab_y_true_tot = torch.cat([ab_y_true_tot, ab_y_true], dim=0).to(self.device)
            positions_y_tot = torch.cat([positions_y_tot, positions_y], dim=0).to(self.device)

        # self.epoch += 1
        if self.scheduler:
            self.scheduler.step()
        return ab_y_hat_tot, ab_y_true_tot, positions_y_tot

    def train(self, n_epochs: int):
        for e in range(n_epochs):
            # start_time = time.time()
            train_loss, acc_tot, me_tot, vme_tot = self._train_epoch()
            # print("--- training: %s seconds ---" % (time.time() - start_time))

            wandb.log({'Training epoch loss': train_loss, "epoch": e})
            wandb.log({'Training epoch accuracy': acc_tot, "epoch": e})
            # wandb.log({'Training epoch major error rate':me_tot, "epoch": e})
            # wandb.log({'Training epoch very major error rate':vme_tot, "epoch": e})

            print(self.epoch, "Train loss: {}".format(train_loss))
            print(self.epoch, "Accuracy: {}".format(acc_tot))
            print(self.epoch, "Major error: {}".format(me_tot))
            print(self.epoch, "Very major error: {}".format(vme_tot))
            if self.val_loader is not None:
                if e % self.eval_every_n == 0:
                    ab_y_hat, ab_y_true, positions_y = self._val_epoch()

                    # print(ab_y_true)
                    # print(positions_y)
                    # print(ab_y_hat)

                    acc_list = [self.acc[i](ab_y_hat[positions_y == i, :], ab_y_true[positions_y == i])
                                for i in range(self.number_ab)]

                    error_rates = [error_rates_2(ab_y_hat[positions_y == i, :], ab_y_true[positions_y == i])
                                   for i in range(self.number_ab)]

                    me_list = [error_rates[i][0] for i in range(len(error_rates))]
                    vme_list = [error_rates[i][1] for i in range(len(error_rates))]

                    weighted_acc = 0
                    weighted_me = 0
                    weighted_vme = 0
                    for i in range(16):
                        wandb.log({'Val cls {} epoch accuracy'.format(abbrev_pos_list[i]): acc_list[i], "epoch": e})
                        weighted_acc += acc_list[i] * self.weights_val[i]
                        weighted_me += me_list[i] * self.weights_s_val[i]
                        weighted_vme += vme_list[i] * self.weights_r_val[i]

                    # print(self.epoch, "Accuracy Val Weighted: {}".format(weighted_acc))
                    # print(self.epoch, "Major error Weighted: {}".format(weighted_me))
                    # print(self.epoch, "Very major error Weighted: {}".format(weighted_vme))

                    wandb.log({'Val epoch accuracy': weighted_acc, "epoch": e})
                    wandb.log({'Val epoch major error rate': weighted_me, "epoch": e})
                    wandb.log({'Val epoch very major error rate': weighted_vme, "epoch": e})
                    '''
                    if e > self.num_epochs - self.eval_every_n - 1:
                        df = pd.read_csv(f"/cephyr/users/oskgus/Alvis/Master thesis/FCC-Alvis/FCCgit/dire/DataFrames/differ_{self.df_details[2]}.csv")

                        df[self.df_details[3]].loc[0] = weighted_acc
                        df[self.df_details[3]].loc[1] = weighted_me
                        df[self.df_details[3]].loc[2] = weighted_vme

                        # we save data in the column corresponding to species, as indicated by df_details[3]
                        df.to_csv(f"/cephyr/users/oskgus/Alvis/Master thesis/FCC-Alvis/FCCgit/dire/DataFrames/differ_{self.df_details[2]}.csv")

                '''