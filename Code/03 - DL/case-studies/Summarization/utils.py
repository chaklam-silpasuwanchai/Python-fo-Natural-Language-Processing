# -*- coding: utf-8 -*-

import os
import logging
import numpy as np

import torch
import tensorflow as tf

from utils import config









def show_art_oovs(article, vocab):
    unk_token = vocab.word2id(config.UNK_TOKEN)
    words = article.split(' ')
    words = [("__%s__" % w) if vocab.word2id(w) == unk_token else w for w in words]
    out_str = ' '.join(words)
    return out_str


def show_abs_oovs(abstract, vocab, article_oovs):
    unk_token = vocab.word2id(config.UNK_TOKEN)
    words = abstract.split(' ')
    new_words = []
    for w in words:
        if vocab.word2id(w) == unk_token:  # w is oov
            if article_oovs is None:  # baseline mode
                new_words.append("__%s__" % w)
            else:  # pointer-generator mode
                if w in article_oovs:
                    new_words.append("__%s__" % w)
                else:
                    new_words.append("!!__%s__!!" % w)
        else:  # w is in-vocab word
            new_words.append(w)
    out_str = ' '.join(new_words)
    return out_str


def print_results(article, abstract, decoded_output):
    print("")
    print('ARTICLE:  %s', article)
    print('REFERENCE SUMMARY: %s', abstract)
    print('GENERATED SUMMARY: %s', decoded_output)
    print("")














def get_input_from_batch(batch, use_cuda):
    extra_zeros = None
    enc_lens = batch.enc_lens
    max_enc_len = np.max(enc_lens)
    enc_batch_extend_vocab = None
    batch_size = len(batch.enc_lens)
    enc_batch = Variable(torch.from_numpy(batch.enc_batch).long())
    enc_padding_mask = Variable(torch.from_numpy(batch.enc_padding_mask)).float()

    if config.pointer_gen:
        enc_batch_extend_vocab = Variable(torch.from_numpy(batch.enc_batch_extend_vocab).long())
        # max_art_oovs is the max over all the article oov list in the batch
        if batch.max_art_oovs > 0:
            extra_zeros = Variable(torch.zeros((batch_size, batch.max_art_oovs)))

    c_t = Variable(torch.zeros((batch_size, 2 * config.hidden_dim)))

    coverage = None
    if config.is_coverage:
        coverage = Variable(torch.zeros(enc_batch.size()))

    enc_pos = np.zeros((batch_size, max_enc_len))
    for i, inst in enumerate(batch.enc_batch):
        for j, w_i in enumerate(inst):
            if w_i != config.PAD:
                enc_pos[i, j] = (j + 1)
            else:
                break
    enc_pos = Variable(torch.from_numpy(enc_pos).long())

    if use_cuda:
        c_t = c_t.cuda()
        enc_pos = enc_pos.cuda()
        enc_batch = enc_batch.cuda()
        enc_padding_mask = enc_padding_mask.cuda()

        if coverage is not None:
            coverage = coverage.cuda()

        if extra_zeros is not None:
            extra_zeros = extra_zeros.cuda()

        if enc_batch_extend_vocab is not None:
            enc_batch_extend_vocab = enc_batch_extend_vocab.cuda()


    return enc_batch, enc_lens, enc_pos, enc_padding_mask, enc_batch_extend_vocab, extra_zeros, c_t, coverage



