# -*- coding: utf-8 -*-

import os
import pyrouge
import logging
import numpy as np

import torch
import tensorflow as tf

from utils import config




def outputids2words(id_list, vocab, article_oovs):
    words = []
    for i in id_list:
        try:
            w = vocab.id2word(i)  # might be [UNK]
        except ValueError as e:  # w is OOV
            assert article_oovs is not None, \
                "Error: models produced a word ID that isn't in the vocabulary. This should not happen in baseline (no pointer-generator) mode"
            article_oov_idx = i - vocab.size()
            try:
                w = article_oovs[article_oov_idx]
            except ValueError as e:  # i doesn't correspond to an article oov
                raise ValueError(
                    'Error: models produced word ID %i which corresponds to article OOV %i but this example only has %i article OOVs' % (
                        i, article_oov_idx, len(article_oovs)))
        words.append(w)
    return words





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


def make_html_safe(s):
    s.replace("<", "&lt;")
    s.replace(">", "&gt;")
    return s


def rouge_eval(ref_dir, dec_dir):
    r = pyrouge.Rouge155()
    r.model_filename_pattern = '#ID#_reference.txt'
    r.system_filename_pattern = '(\d+)_decoded.txt'
    r.model_dir = ref_dir
    r.system_dir = dec_dir
    logging.getLogger('global').setLevel(logging.WARNING)  # silence pyrouge logging
    rouge_results = r.convert_and_evaluate()
    return r.output_to_dict(rouge_results)


def rouge_log(results_dict, dir_to_write):
    log_str = ""
    for x in ["1", "2", "l"]:
        log_str += "\nROUGE-%s:\n" % x
        for y in ["f_score", "recall", "precision"]:
            key = "rouge_%s_%s" % (x, y)
            key_cb = key + "_cb"
            key_ce = key + "_ce"
            val = results_dict[key]
            val_cb = results_dict[key_cb]
            val_ce = results_dict[key_ce]
            log_str += "%s: %.4f with confidence interval (%.4f, %.4f)\n" % (key, val, val_cb, val_ce)
    print(log_str)
    results_file = os.path.join(dir_to_write, "ROUGE_results.txt")
    print("Writing final ROUGE results to %s..." % (results_file))
    with open(results_file, "w") as f:
        f.write(log_str)





def write_for_rouge(reference_sents, decoded_words, ex_index,
                    _rouge_ref_dir, _rouge_dec_dir):
    decoded_sents = []
    while len(decoded_words) > 0:
        try:
            fst_period_idx = decoded_words.index(".")
        except ValueError:
            fst_period_idx = len(decoded_words)
        sent = decoded_words[:fst_period_idx + 1]
        decoded_words = decoded_words[fst_period_idx + 1:]
        decoded_sents.append(' '.join(sent))

    # pyrouge calls a perl script that puts the data into HTML files.
    # Therefore we need to make our output HTML safe.
    decoded_sents = [make_html_safe(w) for w in decoded_sents]
    reference_sents = [make_html_safe(w) for w in reference_sents]

    ref_file = os.path.join(_rouge_ref_dir, "%06d_reference.txt" % ex_index)
    decoded_file = os.path.join(_rouge_dec_dir, "%06d_decoded.txt" % ex_index)

    with open(ref_file, "w") as f:
        for idx, sent in enumerate(reference_sents):
            f.write(sent) if idx == len(reference_sents) - 1 else f.write(sent + "\n")
    with open(decoded_file, "w") as f:
        for idx, sent in enumerate(decoded_sents):
            f.write(sent) if idx == len(decoded_sents) - 1 else f.write(sent + "\n")

    # print("Wrote example %i to file" % ex_index)


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



