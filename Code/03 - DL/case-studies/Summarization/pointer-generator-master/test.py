# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function, division

import os
import sys
import time
import torch
from torch.autograd import Variable

from models.model import Model
from utils import utils
from utils.dataset import Batcher
from utils.dataset import Vocab
from utils import dataset, config
from utils.utils import get_input_from_batch
from utils.utils import write_for_rouge, rouge_eval, rouge_log

use_cuda = config.use_gpu and torch.cuda.is_available()


class Beam(object):
    def __init__(self, tokens, log_probs, state, context, coverage):
        self.tokens = tokens
        self.state = state
        self.context = context
        self.coverage = coverage
        self.log_probs = log_probs

    def extend(self, token, log_prob, state, context, coverage):
        return Beam(tokens=self.tokens + [token],
                    log_probs=self.log_probs + [log_prob],
                    state=state,
                    context=context,
                    coverage=coverage)

    @property
    def latest_token(self):
        return self.tokens[-1]

    @property
    def avg_log_prob(self):
        return sum(self.log_probs) / len(self.tokens)


class BeamSearch(object):
    def __init__(self, model_file_path):
        
        model_name = os.path.basename(model_file_path)
        self._test_dir = os.path.join(config.log_root, 'decode_%s' % (model_name))
        self._rouge_ref_dir = os.path.join(self._test_dir, 'rouge_ref')
        self._rouge_dec_dir = os.path.join(self._test_dir, 'rouge_dec')
        for p in [self._test_dir, self._rouge_ref_dir, self._rouge_dec_dir]:
            if not os.path.exists(p):
                os.mkdir(p)

        self.vocab = Vocab(config.vocab_path, config.vocab_size)
        self.batcher = Batcher(config.decode_data_path, self.vocab, mode='decode',
                               batch_size=config.beam_size, single_pass=True)
        time.sleep(15)

        self.model = Model(model_file_path, is_eval=True)

    def sort_beams(self, beams):
        return sorted(beams, key=lambda h: h.avg_log_prob, reverse=True)
    

    def beam_search(self, batch):
        # single example repeated across the batch
        enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, c_t, coverage = \
            get_input_from_batch(batch, use_cuda)

        enc_out, enc_fea, enc_h = self.model.encoder(enc_batch, enc_lens)
        s_t = self.model.reduce_state(enc_h)

        dec_h, dec_c = s_t     # b x hidden_dim
        dec_h = dec_h.squeeze()
        dec_c = dec_c.squeeze()

        # decoder batch preparation, it has beam_size example initially everything is repeated
        beams = [Beam(tokens=[self.vocab.word2id(config.BOS_TOKEN)],
                      log_probs=[0.0],
                      state=(dec_h[0], dec_c[0]),
                      context=c_t[0],
                      coverage=(coverage[0] if config.is_coverage else None))
                 for _ in range(config.beam_size)]

        steps = 0
        results = []
        while steps < config.max_dec_steps and len(results) < config.beam_size:
            latest_tokens = [h.latest_token for h in beams]
            latest_tokens = [t if t < self.vocab.size() else self.vocab.word2id(config.UNK_TOKEN) \
                             for t in latest_tokens]
            y_t = Variable(torch.LongTensor(latest_tokens))
            if use_cuda:
                y_t = y_t.cuda()
            all_state_h = [h.state[0] for h in beams]
            all_state_c = [h.state[1] for h in beams]
            all_context = [h.context  for h in beams]

            s_t = (torch.stack(all_state_h, 0).unsqueeze(0), torch.stack(all_state_c, 0).unsqueeze(0))
            c_t = torch.stack(all_context, 0)

            coverage_t = None
            if config.is_coverage:
                all_coverage = [h.coverage for h in beams]
                coverage_t = torch.stack(all_coverage, 0)

            final_dist, s_t, c_t, attn_dist, p_gen, coverage_t = self.model.decoder(y_t, s_t,
                                                                                    enc_out, enc_fea,
                                                                                    enc_padding_mask, c_t,
                                                                                    extra_zeros, enc_batch_extend_vocab,
                                                                                    coverage_t, steps)
            log_probs = torch.log(final_dist)
            topk_log_probs, topk_ids = torch.topk(log_probs, config.beam_size * 2)

            dec_h, dec_c = s_t
            dec_h = dec_h.squeeze()
            dec_c = dec_c.squeeze()

            all_beams = []
            # On the first step, we only had one original hypothesis (the initial hypothesis). On subsequent steps, all original hypotheses are distinct.
            num_orig_beams = 1 if steps == 0 else len(beams)
            for i in range(num_orig_beams):
                h = beams[i]
                state_i = (dec_h[i], dec_c[i])
                context_i = c_t[i]
                coverage_i = (coverage[i] if config.is_coverage else None)

                for j in range(config.beam_size * 2):  # for each of the top 2*beam_size hyps:
                    new_beam = h.extend(token=topk_ids[i, j].item(),
                                        log_prob=topk_log_probs[i, j].item(),
                                        state=state_i,
                                        context=context_i,
                                        coverage=coverage_i)
                    all_beams.append(new_beam)

            beams = []
            for h in self.sort_beams(all_beams):
                if h.latest_token == self.vocab.word2id(config.EOS_TOKEN):
                    if steps >= config.min_dec_steps:
                        results.append(h)
                else:
                    beams.append(h)
                if len(beams) == config.beam_size or len(results) == config.beam_size:
                    break

            steps += 1

        if len(results) == 0:
            results = beams

        beams_sorted = self.sort_beams(results)

        return beams_sorted[0]
    
    def run(self):
        
        counter = 0
        start = time.time()
        batch = self.batcher.next_batch()
        while batch is not None:
            # Run beam search to get best Hypothesis
            best_summary = self.beam_search(batch)

            # Extract the output ids from the hypothesis and convert back to words
            output_ids = [int(t) for t in best_summary.tokens[1:]]
            decoded_words = utils.outputids2words(output_ids, self.vocab,
                                                    (batch.art_oovs[0] if config.pointer_gen else None))

            # Remove the [STOP] token from decoded_words, if necessary
            try:
                fst_stop_idx = decoded_words.index(dataset.EOS_TOKEN)
                decoded_words = decoded_words[:fst_stop_idx]
            except ValueError:
                decoded_words = decoded_words

            original_abstract_sents = batch.original_abstracts_sents[0]

            write_for_rouge(original_abstract_sents, decoded_words, counter,
                            self._rouge_ref_dir, self._rouge_dec_dir)
            counter += 1
            if counter % 1000 == 0:
                print('%d example in %d sec' % (counter, time.time() - start))
                start = time.time()

            batch = self.batcher.next_batch()

        print("Decoder has finished reading dataset for single_pass.")
        print("Now starting ROUGE eval...")
        results_dict = rouge_eval(self._rouge_ref_dir, self._rouge_dec_dir)
        rouge_log(results_dict, self._test_dir)


if __name__ == '__main__':
    model_filename = sys.argv[1]
    test_processor = BeamSearch(model_filename)
    test_processor.run()
