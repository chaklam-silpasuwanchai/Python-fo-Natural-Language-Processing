# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function, division

import os
import time
import argparse
import numpy as np
import tensorflow as tf

import torch
import torch.optim as optim

from transformer.model import Model
from transformer.optim import ScheduledOptim
from utils import config
from utils.dataset import Vocab
from utils.dataset import Batcher
from utils.utils import get_input_from_batch
from utils.utils import get_output_from_batch
from utils.utils import calc_running_avg_loss

use_cuda = config.use_gpu and torch.cuda.is_available()

class Train(object):
    def __init__(self):
        self.vocab = Vocab(config.vocab_path, config.vocab_size)
        self.batcher = Batcher(self.vocab, config.train_data_path,
                               config.batch_size, single_pass=False, mode='train')
        time.sleep(10)

        train_dir = os.path.join(config.log_root, 'train_%d' % (int(time.time())))
        if not os.path.exists(train_dir):
            os.mkdir(train_dir)

        self.model_dir = os.path.join(train_dir, 'models')
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)

        self.summary_writer = tf.summary.FileWriter(train_dir)

    def save_model(self, running_avg_loss, iter):
        model_state_dict = self.model.state_dict()

        state = {
            'iter': iter,
            'current_loss': running_avg_loss,
            'optimizer': self.optimizer._optimizer.state_dict(),
            "model": model_state_dict
        }
        model_save_path = os.path.join(self.model_dir, 'model_%d_%d' % (iter, int(time.time())))
        torch.save(state, model_save_path)

    def setup_train(self, model_path):

        device = torch.device('cuda' if use_cuda else 'cpu')

        self.model = Model(
            config.vocab_size,
            config.vocab_size,
            config.max_enc_steps,
            config.max_dec_steps,
            d_k=config.d_k,
            d_v=config.d_v,
            d_model=config.d_model,
            d_word_vec=config.emb_dim,
            d_inner=config.d_inner_hid,
            n_layers=config.n_layers,
            n_head=config.n_head,
            dropout=config.dropout).to(device)

        self.optimizer = ScheduledOptim(
            optim.Adam(
                filter(lambda x: x.requires_grad, self.model.parameters()),
                betas=(0.9, 0.98), eps=1e-09),
            config.d_model, config.n_warmup_steps)


        params = list(self.model.encoder.parameters()) + list(self.model.decoder.parameters())
        total_params = sum([param[0].nelement() for param in params])
        print('The Number of params of model: %.3f million' % (total_params / 1e6))  # million

        start_iter, start_loss = 0, 0

        if model_path is not None:
            state = torch.load(model_path, map_location=lambda storage, location: storage)
            start_iter = state['iter']
            start_loss = state['current_loss']

            if not config.is_coverage:
                self.optimizer._optimizer.load_state_dict(state['optimizer'])
                if use_cuda:
                    for state in self.optimizer._optimizer.state.values():
                        for k, v in state.items():
                            if torch.is_tensor(v):
                                state[k] = v.cuda()

        return start_iter, start_loss

    def train_one_batch(self, batch):
        enc_batch, enc_lens, enc_pos, enc_padding_mask, enc_batch_extend_vocab, \
        extra_zeros, c_t, coverage = get_input_from_batch(batch, use_cuda, transformer=True)
        dec_batch, dec_lens, dec_pos, dec_padding_mask, max_dec_len, tgt_batch = \
            get_output_from_batch(batch, use_cuda, transformer=True)

        self.optimizer.zero_grad()

        pred = self.model(enc_batch, enc_pos, dec_batch, dec_pos)
        gold_probs = torch.gather(pred, -1, tgt_batch.unsqueeze(-1)).squeeze()
        batch_loss = -torch.log(gold_probs + config.eps)
        batch_loss = batch_loss * dec_padding_mask

        sum_losses = torch.sum(batch_loss, 1)
        batch_avg_loss = sum_losses / dec_lens
        loss = torch.mean(batch_avg_loss)

        loss.backward()

        # update parameters
        self.optimizer.step_and_update_lr()

        return loss.item(), 0.

    def run(self, n_iters, model_path=None):
        iter, running_avg_loss = self.setup_train(model_path)
        start = time.time()
        interval = 100

        while iter < n_iters:
            batch = self.batcher.next_batch()
            loss, cove_loss = self.train_one_batch(batch)

            running_avg_loss = calc_running_avg_loss(loss, running_avg_loss, self.summary_writer, iter)
            iter += 1

            if iter % interval == 0:
                self.summary_writer.flush()
                print(
                    'step: %d, second: %.2f , loss: %f, cover_loss: %f' % (iter, time.time() - start, loss, cove_loss))
                start = time.time()
            if iter % 5000 == 0:
                self.save_model(running_avg_loss, iter)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train script")
    parser.add_argument("-m",
                        dest="model_path",
                        required=False,
                        default=None,
                        help="Model file for retraining (default: None).")
    args = parser.parse_args()

    train_processor = Train()
    train_processor.run(config.max_iterations, args.model_path)
