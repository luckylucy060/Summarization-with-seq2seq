# -*- coding: utf-8 -*-

"""
This project is heavily inspired by CS224N Assignment 4
Credit: Standford NLP

Author: Zaixiang Zheng <zaixiang.zheng@gmail.com>
"""

from collections import namedtuple
import sys
from typing import List, Tuple, Dict, Set, Union
import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from modules import LSTMRNN, GlobalAttention

Hypothesis = namedtuple('Hypothesis', ['value', 'score'])

class ModelEmbeddings(nn.Module):
    """
    Class that converts input words to their embeddings.
    """
    def __init__(self, embed_size, vocab):
        """
        Init the Embedding layers.

        @param embed_size (int): Embedding size (dimensionality)
        @param vocab (Vocab): Vocabulary object containing src and tgt languages
                              See vocab.py for documentation.
        """
        super(ModelEmbeddings, self).__init__()
        self.embed_size = embed_size

        # default values
        self.source = nn.Embedding(len(vocab.src), embed_size,
                                   padding_idx=vocab.src["<pad>"])
        self.target = nn.Embedding(len(vocab.tgt), embed_size,
                                   padding_idx=vocab.src["<pad>"])



class BiLSTMEncoder(nn.Module):
    """ Simple Bidirectional LSTM Encoder """
    def __init__(self, embed_size, hidden_size):
        super().__init__()
        self.fwd_rnn = LSTMRNN(embed_size, hidden_size, "LSTM")
        self.bwd_rnn = LSTMRNN(embed_size, hidden_size, "LSTM")

    def forward(self, x, x_mask):
        """
        @param x (torch.FloatTensor): embedding of input sentence. [bsz, L, embed_size]
        @param x_mask (torch.ByteTensor): mask of input sentence. 0s for paddings, otherwise 1s.
        """
        x_mask = x_mask.float()
        fwd_hiddens, _  = self.fwd_rnn(x, x_mask)
        bwd_x, bwd_mask = self.reverse_sequence(x, x_mask)

        bwd_hiddens, _ = self.bwd_rnn(bwd_x, bwd_mask)
        bwd_hiddens, = self.reverse_sequence(bwd_hiddens)

        enc_hiddens = torch.cat([fwd_hiddens, bwd_hiddens], dim=-1)
        return enc_hiddens

    def reverse_sequence(self, *tensors):
        """ Reverse tensors through dimension 1 (the length dim)
        @params: tensors [List[torch.Tensor]] [bsz, L, *]
        """
        rev_tensors = []
        for tensor in tensors:
            L = tensor.size(1)
            index = torch.arange(L-1, -1, -1)
            rev_tensors.append(torch.index_select(tensor, dim=1, index=index))
        return tuple(rev_tensors)


class LSTMDecoder(nn.Module):
    def __init__(self, embed_size, hidden_size):
        super().__init__()
        self.rnn = LSTMRNN(embed_size, hidden_size, "LSTM")
        self.init = nn.Sequential(
            nn.Linear(2*hidden_size, 2*hidden_size),
            nn.Tanh())

    def forward(self, x, init_state=None):
        """
        @param x (torch.FloatTensor): embedding of input sentence. [bsz, L, embed_size]
        @param init_hidden (torch.FloatTensor): embedding of input sentence. [bsz, hidden_size]
        """
        if init_state is None:
            ValueError("Argument 'init_hidden' is required.")
        dec_states = self.rnn(x, mask=None, init_hidden=init_state)

        return dec_states

    def init_decoder(self, enc_hiddens, enc_mask):
        """
        @param enc_hiddens (torch.FloatTensor): hidden states of source sentence. [bsz, L, 2*hidden_size]
        @param enc_mask (torch.ByteTensor): mask of input sentence. 0s for paddings, otherwise 1s. [bsz, L]
        """
        enc_mask = enc_mask.float()
        enc_lens = enc_mask.sum(-1)

        # [bsz, L, 2*hidden_size]
        average_hidden = (enc_hiddens * enc_mask[:, :, None]).sum(1)\
                    / enc_lens[:, None]
        # [bsz, 2*hidden_size]
        dec_init_state = self.init(average_hidden)
        # [bsz, hidden_size]
        init_hidden, init_cell = torch.chunk(dec_init_state, chunks=2, dim=-1)
        dec_init_state = (init_hidden, init_cell)

        return dec_init_state


class NMT(nn.Module):
    """ Simple Neural Machine Translation Model:
        - Bidrectional LSTM Encoder
        - Unidirection LSTM Decoder
        - Global Attention Model (Luong, et al. 2015)
    """
    def __init__(self, embed_size, hidden_size, vocab, dropout_rate=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.vocab = vocab

        # define embeddings, encoder, decoder and attention model
        self.embeddings = ModelEmbeddings(embed_size, vocab)
        self.encoder = BiLSTMEncoder(embed_size, hidden_size)
        self.decoder = LSTMDecoder(embed_size, hidden_size)
        self.attention = GlobalAttention(
            query_size=hidden_size, value_size=2*hidden_size)
        self.combiner = nn.Sequential(
            nn.Linear(3*hidden_size, embed_size),
            nn.Tanh(),
            nn.Dropout(dropout_rate))
        self.target_vocab_projection = nn.Linear(embed_size, len(vocab.tgt), bias=False)


    def forward(self, source: List[List[str]], target: List[List[str]]) -> torch.Tensor:
        """ Take a mini-batch of source and target sentences, compute the log-likelihood of
        target sentences under the language models learned by the NMT system.

        @param source (List[List[str]]): list of source sentence tokens
        @param target (List[List[str]]): list of target sentence tokens, wrapped by `<s>` and `</s>`

        @returns scores (Tensor): a variable/tensor of shape (b, ) representing the log-likelihood of
        generating the gold-standard target sentence for each example in the input batch. Here b = batch size.
        """

        # [bsz, L]

        # 0. Tensorize inputs
        # Convert list of lists into tensors
        # [bsz, Lx]
        source_padded = self.vocab.src.to_input_tensor(source, device=self.device)
        source_mask = self.generate_sent_mask(source_padded, self.vocab.src["<pad>"])

        # [bsz, Ly]
        target_padded = self.vocab.tgt.to_input_tensor(target, device=self.device)   # Tensor: (tgt_len, b)
        # determin target input and label
        # [bsz, Ly-1]
        target_padded_input = target_padded[:, :-1].contiguous()
        target_padded_label = target_padded[:, 1:].contiguous()
        target_label_mask = self.generate_sent_mask(target_padded_label, self.vocab.tgt["<pad>"])

        # 0. Get embeddings
        source_embed = self.embeddings.source(source_padded) 
        target_input_embed = self.embeddings.target(target_padded_input) 

        # 1. Encode source sentence
        enc_hiddens = self.encode(source_embed, source_mask)

        # 2. Decode target sentence from initial decoder
        # hidden state computed by encoder hidden states
        dec_init_state = self.decoder.init_decoder(
            enc_hiddens,
            enc_mask=source_mask)

        decoded_outs = self.decode(target_input_embed, dec_init_state, enc_hiddens, source_mask)

        # 3. Compute logits and log-probs
        # [bsz, Ly-1, len(vocab.tgt)]
        logits = self.target_vocab_projection(decoded_outs)
        log_P = F.log_softmax(logits, dim=-1)

        # Zero out probs of paddings in the target sentence
        gold_log_P = log_P * target_label_mask.unsqueeze(-1).float()

        # 4. Compute scores (negative loss)
        # [bsz, Ly-1]
        # scores = - gold_log_P.sum(dim=1)
        bsz, Ly, _ = gold_log_P.size()
        scores = F.nll_loss(gold_log_P.view(bsz*Ly, -1), target_padded_label.view(-1),
                            ignore_index=self.vocab.tgt["<pad>"],
                            reduce=False)\
                            .view(bsz, Ly).sum(-1)

        return scores

    def encode(self, source_padded, source_mask):
        hiddens = self.encoder(source_padded, source_mask)
        return hiddens

    def decode(self, target_padded, dec_init_state, enc_hiddens, enc_mask):
        # Compute decoder hidden states
        dec_hiddens, dec_cells = self.decoder(target_padded, dec_init_state)
        # Perform attention model
        context_vectors, attention_scores = \
            self.attention(dec_hiddens, enc_hiddens, enc_mask)
        # Combine decoder hidden states and context vectors
        out = self.combiner(torch.cat([dec_hiddens, context_vectors], -1))

        return out 

    def decode_step(self, target_input, dec_last_state, enc_hiddens, enc_mask):
        """
        @param target_input (torch.FloatTensor): [B, 1, embed_size]
        @param dec_last_state (Tuple[Tensor, Tensor]): hidden and cell of last step [B, hidden_size]
        """
        # pretend that we have length=1
        # target_input = target_input.unsqueeze(1)

        dec_hiddens, dec_cells = self.decoder(target_input, dec_last_state)

        # Perform attention model
        context_vectors, attention_scores = self.attention(dec_hiddens, enc_hiddens, enc_mask)
        # Combine decoder hidden states and context vectors
        out = self.combiner(torch.cat([dec_hiddens, context_vectors], -1)) 

        # squeeze
        (out, dec_hiddens, dec_cells) = (t.squeeze(1) for t in [out, dec_hiddens, dec_cells])

        return out, (dec_hiddens, dec_cells)
    
    def beam_search2(self, src_sent: List[str], beam_size: int=5, max_decoding_time_step: int=70) -> List[Hypothesis]:
        """ Given a single source sentence, perform beam search, yielding translations in the target language.
        @param src_sent (List[str]): a single source sentence (words)
        @param beam_size (int): beam size
        @param max_decoding_time_step (int): maximum number of time steps to unroll the decoding RNN
        @returns hypotheses (List[Hypothesis]): a list of hypothesis, each hypothesis has two fields:
                value: List[str]: the decoded target sentence, represented as a list of words
                score: float: the log-likelihood of the target sentence
        """
        src_sents_var = self.vocab.src.to_input_tensor([src_sent], self.device)

        src_encodings, dec_init_vec = self.encode(src_sents_var, [len(src_sent)])
        src_encodings_att_linear = self.att_projection(src_encodings)

        h_tm1 = dec_init_vec
        att_tm1 = torch.zeros(1, self.hidden_size, device=self.device)

        eos_id = self.vocab.tgt['</s>']

        hypotheses = [['<s>']]
        hyp_scores = torch.zeros(len(hypotheses), dtype=torch.float, device=self.device)
        completed_hypotheses = []

        t = 0
        while len(completed_hypotheses) < beam_size and t < max_decoding_time_step:
            t += 1
            hyp_num = len(hypotheses)

            exp_src_encodings = src_encodings.expand(
                hyp_num,
                src_encodings.size(1),
                src_encodings.size(2))

            exp_src_encodings_att_linear = src_encodings_att_linear.expand(
                hyp_num,
                src_encodings_att_linear.size(1),
                src_encodings_att_linear.size(2))

            y_tm1 = torch.tensor([self.vocab.tgt[hyp[-1]]
                                  for hyp in hypotheses], dtype=torch.long, device=self.device)
            y_t_embed = self.model_embeddings.target(y_tm1)

            x = torch.cat([y_t_embed, att_tm1], dim=-1)

            (h_t, cell_t), att_t, _ = self.step(x, h_tm1,
                                                      exp_src_encodings, exp_src_encodings_att_linear, enc_masks=None)

            # log probabilities over target words
            log_p_t = F.log_softmax(self.target_vocab_projection(att_t), dim=-1)

            live_hyp_num = beam_size - len(completed_hypotheses)
            contiuating_hyp_scores = (hyp_scores.unsqueeze(1).expand_as(log_p_t) + log_p_t).view(-1)
            top_cand_hyp_scores, top_cand_hyp_pos = torch.topk(contiuating_hyp_scores, k=live_hyp_num)

            prev_hyp_ids = top_cand_hyp_pos / len(self.vocab.tgt)
            hyp_word_ids = top_cand_hyp_pos % len(self.vocab.tgt)

            new_hypotheses = []
            live_hyp_ids = []
            new_hyp_scores = []

            for prev_hyp_id, hyp_word_id, cand_new_hyp_score in zip(prev_hyp_ids, hyp_word_ids, top_cand_hyp_scores):
                prev_hyp_id = prev_hyp_id.item()
                hyp_word_id = hyp_word_id.item()
                cand_new_hyp_score = cand_new_hyp_score.item()

                hyp_word = self.vocab.tgt.id2word[hyp_word_id]
                new_hyp_sent = hypotheses[prev_hyp_id] + [hyp_word]
                if hyp_word == '</s>':
                    completed_hypotheses.append(Hypothesis(value=new_hyp_sent[1:-1],
                                                           score=cand_new_hyp_score))
                else:
                    new_hypotheses.append(new_hyp_sent)
                    live_hyp_ids.append(prev_hyp_id)
                    new_hyp_scores.append(cand_new_hyp_score)

            if len(completed_hypotheses) == beam_size:
                break

            live_hyp_ids = torch.tensor(live_hyp_ids, dtype=torch.long, device=self.device)
            h_tm1 = (h_t[live_hyp_ids], cell_t[live_hyp_ids])
            att_tm1 = att_t[live_hyp_ids]

            hypotheses = new_hypotheses
            hyp_scores = torch.tensor(new_hyp_scores, dtype=torch.float, device=self.device)

        if len(completed_hypotheses) == 0:
            completed_hypotheses.append(Hypothesis(value=hypotheses[0][1:],
                                                   score=hyp_scores[0].item()))

        completed_hypotheses.sort(key=lambda hyp: hyp.score, reverse=True)

        return completed_hypotheses

    def beam_search(self, src_sent: List[str], beam_size: int=5, max_decoding_time_step: int=70) -> List[Hypothesis]:
        """ Given a single source sentence, perform beam search, yielding translations in the target language.
        @param src_sent (List[str]): a single source sentence (words)
        @param beam_size (int): beam size
        @param max_decoding_time_step (int): maximum number of time steps to unroll the decoding RNN
        @returns hypotheses (List[Hypothesis]): a list of hypothesis, each hypothesis has two fields:
                value: List[str]: the decoded target sentence, represented as a list of words
                score: float: the log-likelihood of the target sentence
        """
        # Tensorize source sentence
        src_sents_var = self.vocab.src.to_input_tensor([src_sent], self.device)
        # Get embedding
        src_embed = self.embeddings.source(src_sents_var)
        src_mask = self.generate_sent_mask(src_sents_var, self.vocab.src["<pad>"])
        # Encode source sentence
        enc_hiddens = self.encode(src_embed, src_mask)
        # Initialize decoder state
        dec_init_state = self.decoder.init_decoder(
            enc_hiddens,
            enc_mask=src_mask)

        eos_id = self.vocab.tgt['</s>']

        hypotheses = [['<s>']]
        hyp_scores = torch.zeros(len(hypotheses), dtype=torch.float, device=self.device)
        completed_hypotheses = []

        dec_last_state = dec_init_state
        t = 0
        while len(completed_hypotheses) < beam_size and t < max_decoding_time_step:
            t += 1
            hyp_num = len(hypotheses)
            exp_enc_hiddens = enc_hiddens.repeat(hyp_num, 1, 1)
            exp_src_mask = src_mask.repeat(hyp_num, 1)

            # Tensorize previous predictions as current input from beam
            y_t = self.vocab.tgt.to_input_tensor(
                [[self.vocab.tgt[hyp[-1]]] for hyp in hypotheses], self.device)
            y_t_embed = self.embeddings.target(y_t)

            # perform one-step decoding
            # [B, d]
            out_t, (h_t, c_t) = self.decode_step(y_t_embed, dec_last_state, exp_enc_hiddens, exp_src_mask)

            # log probabilities over target words
            # [B, n_tgt]
            logit_t = self.target_vocab_projection(out_t)
            log_p_t = F.log_softmax(logit_t, dim=-1)

            # compute log prob. of each hyp
            live_hyp_num = beam_size - len(completed_hypotheses)
            ## score(hyp[1:t]) = score(hyp[1:t-1]) + score(hyp[t])
            contiuating_hyp_scores = (hyp_scores.unsqueeze(1).expand_as(log_p_t) + log_p_t).view(-1)
            ## the topK can be alive. K=remaining beam size
            top_cand_hyp_scores, top_cand_hyp_pos = torch.topk(contiuating_hyp_scores, k=live_hyp_num)
            ## determine the id and current prediction word of the alive hyps
            prev_hyp_ids = top_cand_hyp_pos / len(self.vocab.tgt)
            hyp_word_ids = top_cand_hyp_pos % len(self.vocab.tgt)

            new_hypotheses = []
            live_hyp_ids = []
            new_hyp_scores = []

            for prev_hyp_id, hyp_word_id, cand_new_hyp_score in zip(prev_hyp_ids, hyp_word_ids, top_cand_hyp_scores):
                prev_hyp_id = prev_hyp_id.item()
                hyp_word_id = hyp_word_id.item()
                cand_new_hyp_score = cand_new_hyp_score.item()

                hyp_word = self.vocab.tgt.id2word[hyp_word_id]
                new_hyp_sent = hypotheses[prev_hyp_id] + [hyp_word]
                if hyp_word == '</s>':
                    completed_hypotheses.append(Hypothesis(value=new_hyp_sent[1:-1],
                                                           score=cand_new_hyp_score))
                else:
                    new_hypotheses.append(new_hyp_sent)
                    live_hyp_ids.append(prev_hyp_id)
                    new_hyp_scores.append(cand_new_hyp_score)

            if len(completed_hypotheses) == beam_size:
                break

            live_hyp_ids = torch.tensor(live_hyp_ids, dtype=torch.long, device=self.device)
            dec_last_state = (h_t[live_hyp_ids], c_t[live_hyp_ids])

            hypotheses = new_hypotheses
            hyp_scores = torch.tensor(new_hyp_scores, dtype=torch.float, device=self.device)

        if len(completed_hypotheses) == 0:
            completed_hypotheses.append(Hypothesis(value=hypotheses[0][1:],
                                                   score=hyp_scores[0].item()))

        completed_hypotheses.sort(key=lambda hyp: hyp.score, reverse=True)

        return completed_hypotheses

    def generate_sent_mask(self, input_padded, padding_id):
        """
        @param input_padded (torch.LongTensor): [bsz, L]
        @param padding_idx (int) 
        @returns mask (torch.ByteTensor): mask of input sentence. 0s for paddings, otherwise 1s.
        """
        return input_padded != padding_id 

    @property
    def device(self):
        return self.embeddings.source.weight.device

    @staticmethod
    def load(model_path: str):
        """ Load the model from a file.
        @param model_path (str): path to model
        """
        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        args = params['args']
        model = NMT(vocab=params['vocab'], **args)
        model.load_state_dict(params['state_dict'])

        return model

    def save(self, path: str):
        """ Save the odel to a file.
        @param path (str): path to the model
        """
        print('save model parameters to [%s]' % path, file=sys.stderr)

        params = {
            'args': dict(embed_size=self.embeddings.embed_size, hidden_size=self.hidden_size, dropout_rate=self.dropout_rate),
            'vocab': self.vocab,
            'state_dict': self.state_dict()
        }

        torch.save(params, path)
