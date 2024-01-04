import wandb
import torch
import torch.nn as nn
import numpy as np
import itertools
from tqdm import tqdm
from transformers import get_scheduler
from misc.metrics import error_rates_legacy
from models.ppo import PPO
from torchmetrics.classification import BinaryAccuracy, BinaryRecall, BinaryPrecision, BinarySpecificity


def ast_accuracy_2(predicted_values, real_values):        # also do based on R and S to get very major error rate!
    correct_count = 0
    for i in range(len(real_values)):
        if torch.argmax(predicted_values[i]) == real_values[i]:
            correct_count += 1
    if len(real_values) == 0:
        return 0
    return correct_count/len(real_values)


class IntegratedModelPPO:
    def __init__(self, cfg, pheno_model, geno_model, train_dataloader, eval_dataloader):
        self.num_epochs = cfg['training']['n_epochs']
        self.weights_train = cfg['antibiotics']['train_ab_weights']['all']
        self.weights_val = cfg['antibiotics']['val_ab_weights']['all']
        self.weights_s_val = cfg['antibiotics']['val_ab_weights']['weights_s']
        self.weights_r_val = cfg['antibiotics']['val_ab_weights']['weights_r']
        self.ab_abbrev_list = cfg['antibiotics']['index_list']
        self.known_ab = cfg['data']['known_ab']
        self.known_gene = cfg['data']['known_gene']

        if self.known_ab > 0:
            self.pheno_model = pheno_model
        if self.known_gene > 0:
            self.geno_model = geno_model
        self.device = cfg['device']
        self.number_ab = len(cfg['antibiotics']['antibiotics_in_use'])
        self.abpred = [self.ab_pred(cfg['model']['input_dim'], cfg['model']['hidden_dim']) for _ in range(self.number_ab)]  # 1 neural networks for each ab
        #self.losses = [nn.CrossEntropyLoss(weight=torch.tensor(v, requires_grad=False, dtype=torch.float).to(self.device),
        #                                   reduction='mean') for v in cfg['antibiotics']['res_ratio_train'].values()]
        self.losses = [nn.CrossEntropyLoss(reduction='mean') for v in cfg['antibiotics']['res_ratio_train'].values()]

        #self.acc = [ast_accuracy_2 for _ in range(self.number_ab)]
        self.bacc = BinaryAccuracy().to(self.device)
        self.bspec = BinarySpecificity().to(self.device)
        self.brec = BinaryRecall().to(self.device)
        self.bprec = BinaryPrecision().to(self.device)

        self.train_dataloader = train_dataloader
        #self.mixed_precision = cfg['mixed_precision']
        #self.scaler = torch.cuda.amp.GradScaler(enabled=cfg['mixed_precision'])

        long_param_list = []
        for i in range(len(self.abpred)):
            long_param_list.extend(list(self.abpred[i].parameters()))  # add all parameters from the neural networks
        if self.known_ab > 0:
            long_param_list = long_param_list + list(self.pheno_model.net.parameters())
        if self.known_gene > 0:
            long_param_list = long_param_list + list(self.geno_model.parameters())
        self.optimizer = torch.optim.Adam(long_param_list,
                                          lr=cfg['optimizer']['lr'],
                                          weight_decay=cfg['optimizer']['weight_decay'])
        # optimizer above takes all parameters from both transformers and all neural networks
        self.epoch = 0
        self.train_counter = 0
        self.val_counter = 0
        self.summarywrapper = None
        self.val_loader = eval_dataloader
        self.start_ab_stats = False

        self.n_batches_train = np.ceil(len(self.train_dataloader.dataset) / cfg['data']['train_batch_size'])
        self.n_batches_val = np.ceil(len(self.val_loader.dataset) / cfg['data']['val_batch_size'])
        self.eval_every_n = cfg['training']['val_every_n_epochs']
        self.num_training_steps = self.num_epochs * len(self.train_dataloader)
        self.scheduler = get_scheduler("linear", optimizer=self.optimizer, num_warmup_steps=0,
                                       num_training_steps=self.num_training_steps)

        # PPO Parameters
        self.gene_update_timestep = 2  # update policy every other batch
        self.current_timestep = 0
        K_epochs = 30  # update policy for K epochs
        eps_clip = 0.2  # clip parameter for PPO
        gamma = 0.99  # discount factor
        lr_actor = 0.0003  # learning rate for actor network
        lr_critic = 0.001  # learning rate for critic network
        action_std = None
        has_continuous_action_space = False
        gene_state_dim = self.geno_model.config.hidden_size
        gene_action_dim = self.geno_model.config.vocab_size
        self.ppo_gene_agent = PPO(gene_state_dim, gene_action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,
                                  has_continuous_action_space, self.device, action_std)
        self.gene_current_state = torch.nn.Parameter(torch.randn((1, gene_state_dim), device=self.device),
                                                requires_grad=True)
        self.state_gene_emb = torch.nn.Embedding(gene_action_dim, gene_state_dim, device=self.device,
                                                 padding_idx=self.geno_model.config.pad_token_id)
        self.gene_state_fuser = torch.nn.Sequential(torch.nn.Linear(gene_state_dim * 2, gene_state_dim * 2, device=self.device),
                                                                       torch.nn.ReLU(),
                                                                       torch.nn.Linear(gene_state_dim * 2, gene_state_dim, device=self.device))

    def ab_pred(self, input_dim, hidden_dim):
        self.ab = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                nn.ReLU(), nn.LayerNorm(hidden_dim),
                                nn.Linear(hidden_dim, 2)).to(self.device)
        return self.ab

    def forward(self, input_ids, x, positions_y, batch_index_y, x_valid_len_to,
                attention_mask, gene_ids, training=True):
        if self.known_gene > 0:
            input_ids = input_ids.to(self.device)
            b = input_ids.shape[0]
            attention_mask = attention_mask.to(self.device)

            state = self.gene_current_state
            state = state.repeat(b, 1)
            best_gene = torch.zeros((b, self.known_gene), dtype=torch.int32, device=self.device)

            for j in range(self.known_gene):
                gene = self.ppo_gene_agent.select_action(state, training)
                gene = gene.to(dtype=torch.int32, device=self.device)
                best_gene[:, j] = gene
                gemb = self.state_gene_emb(gene)
                state = self.gene_state_fuser(torch.cat([state, gemb], dim=1))
                if training and j < self.known_gene - 1:
                    reward = [0]*b
                    done = [False]*b
                    self.ppo_gene_agent.buffer.rewards += reward
                    self.ppo_gene_agent.buffer.is_terminals += done

            print("Selected best genes for batch 0 is: {}".format(best_gene[0]))
            print(input_ids[0])

            gene_ids = torch.stack([torch.BoolTensor([j in g for j in b]) for b, g in zip(best_gene, input_ids)], dim=0).to(
                dtype=torch.int32, device=self.device)
            print(gene_ids[0])
            #gene_ids = torch.stack([torch.any(torch.eq(input_ids, g), dim=1) for g in best_gene], dim=1).to(dtype=torch.int32)
            cls_tokens = input_ids[:, 0]
            #bg_batched = best_gene.repeat(b, 1)
            attention_mask = None
            input_ids = torch.cat([cls_tokens.unsqueeze(1), gene_ids], dim=1)


            if torch.isnan(gene_ids).any():
                outputs = self.geno_model(input_ids=input_ids, attention_mask=attention_mask)  # index out of range of self
            else:
                outputs = self.geno_model(input_ids=input_ids, attention_mask=attention_mask, gene_ids=gene_ids)  # index out of range of self
            #output_logits = outputs.logits
            # print("shape of AMR model logits: {}".format(output_logits.shape))
            #cls_tokens_amr = output_logits[:, 0, :]
            # print("shape of AMR model cls token: {}".format(cls_tokens_amr.shape))
            cls_tokens_amr = outputs.last_hidden_state[:, 0, :]

        if self.known_ab > 0:
            x = x.to(self.device)
            x_valid_len_to = x_valid_len_to.to(self.device)
            tokens_ab = self.pheno_model.net.encoder(x, x_valid_len_to)
            # print("shape of AST model tokens: {}".format(tokens_ab.shape))
            cls_tokens_ab = tokens_ab[:, 0, :]
            # print("shape of AST model cls token: {}".format(cls_tokens_ab.shape))

            # other_tokens_amr = output_logits[:,1:,:]
            # other_tokens_ab = tokens_ab[:,1:,:]
            # avg_other_amr = other_tokens_amr.mean(dim=1)
            # avg_other_ab = other_tokens_ab.mean(dim=1)

        if self.known_gene > 0 and self.known_ab > 0:
            # encoded_x = torch.cat([cls_tokens_amr, cls_tokens_ab, avg_other_amr, avg_other_ab], dim = 1).to(self.device)
            encoded_x = torch.cat([cls_tokens_amr, cls_tokens_ab], dim=1).to(self.device)
        elif self.known_gene > 0:
            encoded_x = cls_tokens_amr.to(self.device)
        elif self.known_ab > 0:
            encoded_x = cls_tokens_ab.to(self.device)


        encoded_x = encoded_x[:, None, :]
        # print("shape of encoded x: {}".format(encoded_x.shape))

        mlm_hat = torch.cat([a(encoded_x) for a in self.abpred], 1).to(self.device)
        ab_y_hat = mlm_hat[batch_index_y, positions_y]
        #ab_x_hat = mlm_hat[batch_index_x, positions_x]
        return ab_y_hat, mlm_hat

    def predict(self, input_ids, x, y_pos_antibiotic, len_y, total_len_x,
                attention_mask, gene_ids, training=True):
        # Collect for y prediction
        tmp = [[ly] * len_y[ly] for ly in range(len(len_y))]
        batch_index_y = torch.tensor(list(itertools.chain(*tmp))).to(self.device)
        tmp = torch.reshape(y_pos_antibiotic, (-1,))
        positions_y = tmp[tmp >= 0].to(self.device)

        '''
        # Collect for x prediction
        tmp = [[lx] * len_x[lx] for lx in range(len(len_x))]
        batch_index_x = torch.tensor(list(itertools.chain(*tmp))).to(self.device)
        tmp = torch.reshape(x_pos_antibiotic, (-1,))
        positions_x = tmp[tmp >= 0].to(self.device)
        tmp = torch.reshape(x_resp, (-1,))
        ab_x_true = tmp[tmp >= 0].to(self.device)
        '''

        ab_y_hat, mlm_hat = self.forward(input_ids, x, positions_y, batch_index_y, total_len_x, attention_mask, gene_ids, training)

        return ab_y_hat, positions_y, mlm_hat

    def forward_pass_loss(self, input_ids, x, y_pos_antibiotic, y_resp, len_y,
                          total_len_x, attention_mask, gene_ids, training=True):
        tmp = torch.reshape(y_resp, (-1,))
        ab_y_true = tmp[tmp >= 0].to(self.device)
        ab_y_hat, positions_y, mlm_hat = self.predict(input_ids, x, y_pos_antibiotic, len_y, total_len_x,
                                                            attention_mask, gene_ids, training)

        return ab_y_hat, ab_y_true, positions_y, mlm_hat

    def train_pass_and_loss(self, input_ids, x, y_pos_antibiotic, y_resp, len_y,
                            total_len_x, attention_mask, gene_ids):
        ab_y_hat, ab_y_true, positions_y, mlm_hat = self.forward_pass_loss(input_ids, x, y_pos_antibiotic, y_resp, len_y,
                                                                     total_len_x, attention_mask, gene_ids)

        index_list = []
        for i in range(len(positions_y)):
            if len(ab_y_hat[positions_y == i]) > 0:
                index_list.append(i)
        loss_y = [self.losses[i](ab_y_hat[positions_y == i, :], ab_y_true[positions_y == i])
                  for i in range(self.number_ab) if len(ab_y_true[positions_y == i]) > 0]

        #loss_x = [self.losses[i](ab_x_hat[positions_x == i, :], ab_x_true[positions_x == i])
        #          for i in range(self.number_ab)]

        #acc = [self.acc[i](ab_y_hat[positions_y == i, :], ab_y_true[positions_y == i])
        #       for i in range(self.number_ab)]
        ab_y_hat_amax = torch.argmax(ab_y_hat, dim=1)
        acc = [self.bacc(ab_y_hat_amax[positions_y == i], ab_y_true[positions_y == i]) for i in range(self.number_ab) if len(ab_y_true[positions_y == i]) > 0]
        spec = [self.bspec(ab_y_hat_amax[positions_y == i], ab_y_true[positions_y == i]) for i in range(self.number_ab) if len(ab_y_true[positions_y == i]) > 0]
        rec = [self.brec(ab_y_hat_amax[positions_y == i], ab_y_true[positions_y == i]) for i in range(self.number_ab) if len(ab_y_true[positions_y == i]) > 0]
        prec = [self.bprec(ab_y_hat_amax[positions_y == i], ab_y_true[positions_y == i]) for i in range(self.number_ab) if len(ab_y_true[positions_y == i]) > 0]

        #error_rates = [error_rates_legacy(ab_y_hat[positions_y == i, :], ab_y_true[positions_y == i])
        #               for i in range(self.number_ab) if len(ab_y_true[positions_y == i]) > 0]

        #me = [error_rates[i][0] for i in range(len(error_rates))]
        #vme = [error_rates[i][1] for i in range(len(error_rates))]

        loss_y = sum(loss_y) / len(loss_y)
        #loss_x = sum(loss_x) / len(loss_x)
        #acc = sum(acc) / len(index_list)
        #me = sum(me) / len(index_list)
        #vme = sum(vme) / len(index_list)

        acc = sum(acc) / len(acc)
        spec = sum(spec) / len(spec)
        rec = sum(rec) / len(rec)
        prec = sum(prec) / len(prec)
        me = 1 - spec
        vme = 1 - rec

        rewards, acc2 = self.compute_rewards(mlm_hat, y_resp, y_pos_antibiotic)
        acc2 = sum(acc2) / len(acc2)
        #print("reward len: {}".format(len(rewards)))
        terminals = [True]*len(rewards)
        self.ppo_gene_agent.buffer.rewards += rewards
        self.ppo_gene_agent.buffer.is_terminals += terminals
        self.current_timestep += 1
        if self.current_timestep % self.gene_update_timestep == 0:
            print("Updating PPO agent with a buffer of size {}".format(len(self.ppo_gene_agent.buffer.states)))
            self.ppo_gene_agent.update()
            print("Update complete")

        return ab_y_hat, ab_y_true, loss_y, acc, me, vme, prec, acc2

    def val_pass_and_loss(self, input_ids, x, y_pos_antibiotic, y_resp, len_y,
                          total_len_x, attention_mask, gene_ids):
        with torch.no_grad():
            ab_y_hat, ab_y_true, positions_y, _ =  self.forward_pass_loss(input_ids, x, y_pos_antibiotic, y_resp, len_y,
                                       total_len_x, attention_mask, gene_ids, training=False)

            loss_y = [self.losses[i](ab_y_hat[positions_y == i, :], ab_y_true[positions_y == i])
                      for i in range(self.number_ab) if len(ab_y_true[positions_y == i]) > 0]

            #loss_x = [self.losses[i](ab_x_hat[positions_x == i, :], ab_x_true[positions_x == i])
            #          for i in range(self.number_ab) if len(ab_x_hat[positions_x == i]) > 0]

            #acc_list = [self.acc[i](ab_y_hat[positions_y == i, :], ab_y_true[positions_y == i])
            #            for i in range(self.number_ab)]

            #error_rates = [error_rates_legacy(ab_y_hat[positions_y == i, :], ab_y_true[positions_y == i])
            #               for i in range(self.number_ab)]

            #me_list = [error_rates[i][0] for i in range(len(error_rates))]
            #vme_list = [error_rates[i][1] for i in range(len(error_rates))]

            #if self.start_ab_stats == True and val == True:
            #    ab_spec_stat(index_list, s_index, r_index, acc_list, me_list, vme_list)

            loss_y = sum(loss_y) / len(loss_y)
            #loss_x = sum(loss_x) / len(loss_x)
            # acc = sum(acc_list)/len(index_list)
            # me = sum(me_list)/len(s_index)
            # vme = sum(vme_list)/len(r_index)
            return ab_y_hat, ab_y_true, loss_y, positions_y

    def _train_epoch(self):
        # Set Train Mode
        # Hur fick vi batch från början?
        if self.known_ab > 0:
            self.pheno_model.net.train()
        if self.known_gene > 0:
            self.geno_model.train()

        train_loss = 0.
        acc_tot = 0
        acc2_tot = 0
        me_tot = 0
        vme_tot = 0
        prec_tot = 0
        print_every_n_batches = 10
        curr_batch = 0
        # for input_ids, x, x_pos_antibiotic, y_pos_antibiotic, x_resp, y_resp, len_x, len_y, total_len_x, attention, labels in self.train_loader:
        for batch in tqdm(self.train_dataloader):
            #with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            batch = {k: v.to(self.device) for k, v in batch.items()}
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
            gene_ids = batch['gene_ids']
            self.optimizer.zero_grad()
            ab_y_hat, ab_y_true, loss_y, acc, me, vme, prec, acc2 = \
                self.train_pass_and_loss(input_ids, x, y_pos_antibiotic, y_resp, len_y,
                                         total_len_x, attention, gene_ids)

            loss = loss_y  # + loss_x

            # Backward Pass
            loss.backward()
            #self.scaler.scale(loss).backward()
            train_loss += loss.item()

            acc_tot += acc
            acc2_tot += acc2
            me_tot += me
            vme_tot += vme
            prec_tot += prec

            '''
            if curr_batch % print_every_n_batches == 0:
                #print(batch)

                print(curr_batch, "Train loss: {}".format(loss.item()))
                print(curr_batch, "Accuracy: {}".format(acc))
                print(curr_batch, "Major error: {}".format(me))
                print(curr_batch, "Very major error: {}".format(vme))
                print(curr_batch, "Precision: {}".format(prec))

                #print("ab y pred: {}".format(ab_y_hat))
                #print("ab x pred: {}".format(ab_y_hat))
                #print("ab y true: {}".format(ab_y_hat))
                #print("ab x true: {}".format(ab_y_hat))
            '''
            curr_batch += 1


            n_batch = self.n_batches_train
            self.train_counter += 1
            # Update the Weights
            self.optimizer.step()
            #self.scaler.step(self.optimizer)
            #self.scaler.update()
        self.epoch += 1
        if self.scheduler:
            self.scheduler.step()
        return train_loss / self.n_batches_train, acc_tot / self.n_batches_train, me_tot / self.n_batches_train, vme_tot / self.n_batches_train, acc2_tot / self.n_batches_train

    def _val_epoch(self):
        # Set Train Mode
        # Hur fick vi batch från början?
        if self.known_ab > 0:
            self.pheno_model.net.eval()
        if self.known_gene > 0:
            self.geno_model.eval()

        val_loss = 0.
        acc_tot = 0
        me_tot = 0
        vme_tot = 0

        ab_y_hat_tot = torch.empty(0).to(self.device)
        ab_y_true_tot = torch.empty(0).to(self.device)
        positions_y_tot = torch.empty(0).to(self.device)
        # for input_ids, x, x_pos_antibiotic, y_pos_antibiotic, x_resp, y_resp, len_x, len_y, total_len_x, attention, labels in self.train_loader:
        for batch in tqdm(self.val_loader):
            batch = {k: v.to(self.device) for k, v in batch.items()}
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
            gene_ids = batch['gene_ids']
            # self.optimizer.zero_grad()
            ab_y_hat, ab_y_true, loss_y, positions_y = \
                self.val_pass_and_loss(input_ids, x, y_pos_antibiotic, y_resp, len_y,
                                       total_len_x, attention, gene_ids)
            # loss = loss_y  # + loss_x

            # Backward Pass
            # loss.backward()
            val_loss += loss_y.item()

            ab_y_hat_tot = torch.cat([ab_y_hat_tot, ab_y_hat], dim=0).to(self.device)
            ab_y_true_tot = torch.cat([ab_y_true_tot, ab_y_true], dim=0).to(self.device)
            positions_y_tot = torch.cat([positions_y_tot, positions_y], dim=0).to(self.device)

        # self.epoch += 1
        if self.scheduler:
            self.scheduler.step()
        return ab_y_hat_tot, ab_y_true_tot, positions_y_tot, val_loss

    def train(self, n_epochs: int):
        for e in range(n_epochs):
            # start_time = time.time()
            train_loss, acc_tot, me_tot, vme_tot, acc2_tot = self._train_epoch()
            # print("--- training: %s seconds ---" % (time.time() - start_time))
            if wandb.run is not None:
                wandb.log({'Training epoch loss': train_loss, "epoch": e})
                wandb.log({'Training epoch accuracy': acc_tot, "epoch": e})
                wandb.log({'Training epoch major error rate':me_tot, "epoch": e})
                wandb.log({'Training epoch very major error rate':vme_tot, "epoch": e})

            print(self.epoch, "Train loss: {}".format(train_loss))
            print(self.epoch, "Accuracy: {}".format(acc_tot))
            print(self.epoch, "Accuracy2: {}".format(acc2_tot))
            print(self.epoch, "Major error: {}".format(me_tot))
            print(self.epoch, "Very major error: {}".format(vme_tot))

            if self.val_loader is not None:
                if e % self.eval_every_n == 0:
                    ab_y_hat, ab_y_true, positions_y, val_loss = self._val_epoch()

                    #acc_list = [self.acc[i](ab_y_hat[positions_y == i, :], ab_y_true[positions_y == i])
                    #            for i in range(self.number_ab)]

                    ab_y_hat_amax = torch.argmax(ab_y_hat, dim=1)
                    acc_list = [self.bacc(ab_y_hat_amax[positions_y == i], ab_y_true[positions_y == i]) if len(ab_y_true[positions_y == i]) > 0 else 0 for i in
                            range(self.number_ab)]
                    me_list = [1 - self.bspec(ab_y_hat_amax[positions_y == i], ab_y_true[positions_y == i]) if len(ab_y_true[positions_y == i]) > 0 else 1 for i in
                            range(self.number_ab)]
                    vme_list = [1 - self.brec(ab_y_hat_amax[positions_y == i], ab_y_true[positions_y == i]) if len(ab_y_true[positions_y == i]) > 0 else 1 for i in
                           range(self.number_ab)]
                    prec_list = [self.bprec(ab_y_hat_amax[positions_y == i], ab_y_true[positions_y == i]) if len(ab_y_true[positions_y == i]) > 0 else 0 for i in
                            range(self.number_ab)]

                    #error_rates = [error_rates_legacy(ab_y_hat[positions_y == i, :], ab_y_true[positions_y == i])
                    #               for i in range(self.number_ab)]

                    #me_list = [error_rates[i][0] for i in range(len(error_rates))]
                    #vme_list = [error_rates[i][1] for i in range(len(error_rates))]

                    weighted_acc = 0
                    weighted_me = 0
                    weighted_vme = 0
                    weighted_prec = 0
                    for i in range(16):
                        if wandb.run is not None:
                            wandb.log({'Val cls {} epoch accuracy'.format(self.ab_abbrev_list[i]['abbrev']): acc_list[i], "epoch": e})
                        weighted_acc += acc_list[i] * self.weights_val[i]
                        weighted_me += me_list[i] * self.weights_s_val[i]
                        weighted_vme += vme_list[i] * self.weights_r_val[i]


                    if wandb.run is not None:
                        wandb.log({'Val epoch accuracy': weighted_acc, "epoch": e})
                        wandb.log({'Val epoch major error rate': weighted_me, "epoch": e})
                        wandb.log({'Val epoch very major error rate': weighted_vme, "epoch": e})

                    print("Val epoch {} weighted acc: {}".format(self.epoch, weighted_acc))
                    print("Val epoch {} weighted major error: {}".format(self.epoch, weighted_me))
                    print("Val epoch {} weighted very major error rate: {}".format(self.epoch, weighted_vme))

    def compute_rewards(self, pred, labels, positions_y):
        pred_amax = torch.argmax(pred, dim=2)
        preds_per_batch = [[[pred_amax[b, i], labels[b, positions_y[b] == i].squeeze()] for i in range(self.number_ab) if i in positions_y[b]] for b in range(pred_amax.shape[0])]
        #print("ppb: {}".format(preds_per_batch[0]))
        #print("len ppb 0: {}".format(len(preds_per_batch[0])))
        tp_fp_tn_fn = [self.compute_tp_fp_tn_fn(i) for i in preds_per_batch]
        #print("tftf: {}".format(tp_fp_tn_fn[0]))

        recall = [tp/(tp + fn) if (tp + fn) > 0 else 0 for tp, fp, tn, fn in tp_fp_tn_fn]
        precision = [tp / (tp + fp) if (tp + fp) > 0 else 0 for tp, fp, tn, fn in tp_fp_tn_fn]
        specificity = [tn/(tn + fp) if (tn + fp) > 0 else 0 for tp, fp, tn, fn in tp_fp_tn_fn]
        acc = [(tp + tn)/(tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0 for tp, fp, tn, fn in tp_fp_tn_fn]
        f1 = [2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0 for prec, rec in zip(precision, recall)]
        #print("f1: {}".format(f1[0]))
        rewards = [f1_b**2 for f1_b in f1]
        return rewards, acc

    def compute_tp_fp_tn_fn(self, ab_pred_true):
        tp = sum([p == 1 and l == 1 for p, l in ab_pred_true])
        fp = sum([p == 1 and l == 0 for p, l in ab_pred_true])
        tn = sum([p == 0 and l == 0 for p, l in ab_pred_true])
        fn = sum([p == 0 and l == 1 for p, l in ab_pred_true])

        return [tp, fp, tn, fn]


