import torch
from torch import nn
import time
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import os
from abc import ABC, abstractmethod
import itertools
import numpy as np
from torch import optim
from torch.utils.data import DataLoader
from typing import List
import pandas as pd


class Model(ABC):
    @abstractmethod
    def predict(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)


class AntibioticModel(Model):

    def __init__(self, net, device):
        self.net = net
        self.device = device

    def save_model(self, modelname: str):
        if not os.path.exists("saved_models"):
            os.mkdir("saved_models")
        torch.save(self.net.state_dict(), "saved_models/{}.pt".format(modelname))

    def load_model(self, pt_weights: str):
        self.net.load_state_dict(torch.load(pt_weights, map_location=self.device))

    def predict(self, x, x_pos_antibiotic, y_pos_antibiotic, x_resp, len_x, len_y, total_len_x, deterministic=False):

        if deterministic:
            self.net.eval()

        # Collect for y prediction
        x = x.to(self.device)
        total_len_x = total_len_x.to(self.device)
        tmp = [[ly]*len_y[ly] for ly in range(len(len_y))]
        batch_index_y = torch.tensor(list(itertools.chain(*tmp))).to(self.device)
        tmp = torch.reshape(y_pos_antibiotic, (-1,))
        positions_y = tmp[tmp >= 0].to(self.device)

        # Collect for x prediction
        tmp = [[lx]*len_x[lx] for lx in range(len(len_x))]
        batch_index_x = torch.tensor(list(itertools.chain(*tmp))).to(self.device)
        tmp = torch.reshape(x_pos_antibiotic, (-1,))
        positions_x = tmp[tmp >= 0].to(self.device)
        tmp = torch.reshape(x_resp, (-1,))
        ab_x_true = tmp[tmp >= 0].to(self.device)

        ab_y_hat, ab_x_hat = self.net(x, positions_y, batch_index_y, positions_x, batch_index_x, total_len_x)
        return ab_y_hat, ab_x_hat, ab_x_true, positions_y, positions_x, batch_index_y, batch_index_x

    def forward_pass_loss(self, x, x_pos_antibiotic, y_pos_antibiotic, x_resp, y_resp, len_x, len_y, total_len_x,
                          deterministic=False):
        tmp = torch.reshape(y_resp, (-1,))
        ab_y_true = tmp[tmp >= 0].to(self.device)
        ab_y_hat, ab_x_hat, ab_x_true, positions_y, positions_x, batch_index_y, batch_index_x = \
            self.predict(x, x_pos_antibiotic, y_pos_antibiotic, x_resp, len_x, len_y, total_len_x, deterministic)

        return ab_y_hat, ab_x_hat, ab_y_true, ab_x_true, positions_y, positions_x, batch_index_y, batch_index_x


class AntibioticModelTrain(AntibioticModel):

    def __init__(self,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 net: nn.Module,
                 losses: List[nn.CrossEntropyLoss],
                 device,
                 optimizer: optim,
                 batch_size: int,
                 scheduler: optim.lr_scheduler = None,
                 number_ab: int = 16,
                 eval_every_n: int = 1,
                 saved_model_name: str = None):

        super(AntibioticModelTrain, self).__init__(net, device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.losses = losses
        self.epoch = 0
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.number_ab = number_ab
        self.batch_size = batch_size
        self.n_batches_train = np.ceil(len(self.train_loader.dataset) / self.batch_size)
        if self.val_loader is not None:
            self.n_batches_val = np.ceil(len(self.val_loader.dataset) / self.batch_size)
        self.eval_every_n = eval_every_n
        self.current_best_val_loss = np.inf
        self.saved_model_name = saved_model_name

    def forward_pass_and_loss(self, x, x_pos_antibiotic, y_pos_antibiotic, x_resp, y_resp, len_x, len_y, total_len_x):
        ab_y_hat, ab_x_hat, ab_y_true, ab_x_true, positions_y, positions_x, _, _ = \
            self.forward_pass_loss(x, x_pos_antibiotic, y_pos_antibiotic, x_resp, y_resp, len_x, len_y, total_len_x)

        loss_y = [self.losses[i](ab_y_hat[positions_y == i, :], ab_y_true[positions_y == i])
                  for i in range(self.number_ab) if len(ab_y_true[positions_y == i]) > 0]
        loss_x = [self.losses[i](ab_x_hat[positions_x == i, :], ab_x_true[positions_x == i])
                  for i in range(self.number_ab) if len(ab_x_hat[positions_x == i]) > 0]

        loss_y = sum(loss_y)/len(loss_y)
        loss_x = sum(loss_x)/len(loss_x)
        return ab_y_hat, ab_x_hat, ab_y_true, ab_x_true, loss_y, loss_x

    def _train_epoch(self):
        # Set Train Mode
        self.net.train()

        train_loss = 0.
        for x, x_pos_antibiotic, y_pos_antibiotic, x_resp, y_resp, len_x, len_y, total_len_x in self.train_loader:
            self.optimizer.zero_grad()
            ab_y_hat, ab_x_hat, ab_y_true, ab_x_true, loss_y, loss_x = \
                self.forward_pass_and_loss(x, x_pos_antibiotic, y_pos_antibiotic, x_resp, y_resp, len_x, len_y,
                                           total_len_x)
            loss = loss_y  # + loss_x

            # Backward Pass
            loss.backward()
            train_loss += loss.item()

            # Update the Weights
            self.optimizer.step()
        self.epoch += 1
        if self.scheduler:
            self.scheduler.step()
        return train_loss / self.n_batches_train

    def train(self, n_epochs: int):
        for e in range(n_epochs):
            # start_time = time.time()
            train_loss = self._train_epoch()
            # print("--- training: %s seconds ---" % (time.time() - start_time))
            print(self.epoch, "Train loss: {}".format(train_loss))
            if self.val_loader is not None:
                if e % self.eval_every_n == 0:
                    # start_time = time.time()
                    val_loss = self.validation()
                    # print("--- validation: %s seconds ---" % (time.time() - start_time))
                    print(self.epoch, "Val loss: {}".format(val_loss))
                    if self.saved_model_name is not None and val_loss < self.current_best_val_loss:
                        self.current_best_val_loss = val_loss
                        self.save_model(self.saved_model_name)

    def validation(self):
        self.net.eval()
        val_loss = 0.
        for x, x_pos_antibiotic, y_pos_antibiotic, x_resp, y_resp, len_x, len_y, total_len_x in self.val_loader:
            self.optimizer.zero_grad()
            ab_y_hat, ab_x_hat, ab_y_true, ab_x_true, loss_y, loss_x = \
                self.forward_pass_and_loss(x, x_pos_antibiotic, y_pos_antibiotic, x_resp, y_resp, len_x, len_y, total_len_x)
            loss = loss_y  # + loss_x
            val_loss += loss.item()

        return val_loss / self.n_batches_val


class AntibioticModelEval(AntibioticModel):

    def __init__(self,
                 net: nn.Module,
                 devices,
                 x_interval: List[int],
                 antibiotics: List[str],
                 threshold: int = 0.5):

        super(AntibioticModelEval, self).__init__(net, devices)
        self.x_interval = x_interval
        self.antibiotics = antibiotics
        self.antibiotics_array = np.array(self.antibiotics)
        self.threshold = threshold
        self.softmax = nn.Softmax(dim=1)

    def evaluation(self, eval_loader: DataLoader):
        interval_len = self.x_interval[1] - self.x_interval[0] + 1
        results = pd.DataFrame(data=np.zeros((len(self.antibiotics)*interval_len, 8), dtype=int),
                               columns=['Antibiotic', 'k', 'TP', 'TN', 'FP', 'FN', 'Unused', 'Used'])
        results['Antibiotic'] = np.repeat(self.antibiotics, interval_len)
        results['k'] = np.tile(np.arange(self.x_interval[0], self.x_interval[1]+1), len(self.antibiotics))
        results = results.set_index(['Antibiotic', 'k'])

        for x, x_pos_antibiotic, y_pos_antibiotic, x_resp, y_resp, len_x, len_y, total_len_x in eval_loader:
            ab_y_hat, _, ab_y_true, _, positions_y, _, _, _ = self.forward_pass_loss(x, x_pos_antibiotic,
                                                                                     y_pos_antibiotic, x_resp,
                                                                                     y_resp, len_x, len_y, total_len_x)
            y_soft = self.softmax(ab_y_hat)
            x_repeats = torch.repeat_interleave(len_x, len_y)
            for i, ab in enumerate(self.antibiotics):
                for k in range(self.x_interval[0], self.x_interval[1]+1):
                    mask = torch.logical_and(positions_y == i, x_repeats.to(self.device) == k)
                    if torch.sum(mask) > 0:
                        y_soft_i = y_soft[mask]
                        y_true = ab_y_true[mask]
                        n = y_soft_i.shape[0]
                        threshold_idx = torch.any(y_soft_i >= self.threshold, dim=1)
                        y_soft_i = y_soft_i[threshold_idx].max(dim=1).indices
                        y_true = y_true[threshold_idx]
                        results.loc[(ab, k), 'Unused'] += n - y_soft_i.shape[0]
                        results.loc[(ab, k), 'Used'] += y_soft_i.shape[0]
                        for pos, j in zip([['TN', 'FN'], ['TP', 'FP']], [0, 1]):
                            mask = y_soft_i == j
                            n_correct = torch.sum(torch.logical_and(mask, y_true == j)).item()
                            n_wrong = torch.sum(mask).item() - n_correct
                            results.loc[(ab, k), pos[0]] += n_correct
                            results.loc[(ab, k), pos[1]] += n_wrong
        return results

    def all_combination_model_scoring(self, eval_loader: DataLoader, comb: tuple):
        k = len(comb)
        n_preds = len(self.antibiotics) - k
        cols = ['comb', 'pred'] + self.antibiotics
        df = pd.DataFrame(data=np.zeros((len(eval_loader.dataset), len(cols)), dtype=float), columns=cols)
        df['comb'] = [' '.join(comb) for _ in range(df.shape[0])]
        pred = list(set(self.antibiotics) - set(comb))
        df['pred'] = [pred for _ in range(df.shape[0])]
        start = 0
        for x, x_pos_antibiotic, y_pos_antibiotic, x_resp, len_x, len_y, total_len_x in eval_loader:
            ab_y_hat, _, _, positions_y, positions_x, _, _ = self.predict(x, x_pos_antibiotic, y_pos_antibiotic,
                                                                          x_resp, len_x, len_y, total_len_x, True)
            stop = start + eval_loader.batch_size - 1
            df.loc[start:stop, self.antibiotics_array[x_pos_antibiotic[0, :k]]] = x_resp[:, :k]
            y_soft = self.softmax(ab_y_hat).detach().cpu()[:, 1].numpy()
            y_pos = positions_y[:n_preds].cpu()
            for i, yp in enumerate(y_pos):
                df.loc[start:stop, self.antibiotics_array[yp]] = y_soft[i::n_preds]

            start = stop + 1
        return df
