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


class LSTMCell(nn.Module):
    def __init__(self, d_embed, d_hidden):
        super().__init__()
        self.d_hidden = d_hidden
        self.d_embed = d_embed
        self.Wf = nn.Linear(d_embed+d_hidden, d_hidden)
        self.Wi = nn.Linear(d_embed+d_hidden, d_hidden)
        self.Wo = nn.Linear(d_embed+d_hidden, d_hidden)
        self.U = nn.Linear(d_embed+d_hidden, d_hidden)

    def forward(self, x, h, c):
        f = torch.sigmoid(self.Wf(torch.cat([x, h], dim=-1)))
        i = torch.sigmoid(self.Wi(torch.cat([x, h], dim=-1)))
        o = torch.sigmoid(self.Wo(torch.cat([x, h], dim=-1)))

        c_hat = torch.tanh(self.U(torch.cat([x, h], dim=-1)))

        new_c = f * c + i * c_hat
        new_h = o * torch.tanh(new_c)

        return new_h, new_c

    def init_hidden(self):
        return torch.zeros((self.d_hidden)), torch.zeros((self.d_hidden))


class LSTMRNN(nn.Module):
    def __init__(self, d_embed, d_hidden, cell_type="LSTM"):
        super().__init__()
        self.cell = {"LSTM": LSTMCell}[cell_type](d_embed, d_hidden)

    def forward(self, x, mask=None, init_hidden: Tuple=None):
        """
        @param x (torch.FloatTensor): embeddings of input sequence [bsz, L, d_emb]
        @param mask (torch.ByteTensor): mask of x. 0s for paddings otherwise 1s.
        @param init_hidden (Tuple[torch.FloatTensor, Torch.FloatTensor]): 
            initial hidden state and LSTM cell. [bsz, d_hidden]
        """
        h, c = [], []
        bsz = x.size(0)
        if mask is None:
            mask = torch.ones(*x.size()[:2], dtype=torch.float32)
        mask = mask.float()
        if init_hidden is None:
            init_hidden = self.init_hidden()
            init_hidden = (t[None, :].repeat(bsz, 1) for t in init_hidden)
        last_h, last_c = init_hidden
        for x_t, m_t in zip(torch.split(x, 1, dim=1), torch.split(mask, 1, dim=1)):
            # x_t [batch, 1, d_emb], m_t [batch, 1]
            x_t = x_t.squeeze(1)  # [batch, 1, d_emb] -> [batch, d_emb]
            h_t, c_t = self.cell(x_t, last_h, last_c)
            # mask padding
            h_t, c_t = h_t * m_t, c_t * m_t
            h.append(h_t); c.append(c_t)
            last_h = h_t; last_c = c_t
        h = torch.stack(h, dim=1)  # [batch_size, len, d_hidden]
        c = torch.stack(c, dim=1)  # [batch_size, len, d_hidden]
        return h, c

    def init_hidden(self):
        return self.cell.init_hidden()


class GlobalAttention(nn.Module):
    """ Global Attention using a bilinear dot-product. Luong et al (2015) """

    def __init__(self, query_size, value_size):
        super().__init__()
        self.query_proj = nn.Linear(query_size, value_size, bias=False)
        
    def forward(self, query, value, value_mask):
        """
        @param query (torch.FloatTensor): [bsz, Lq, q_size]
        @param value (torch.FloatTensor): [bsz, Lv, v_size]
        @param value_mask (torch.ByteTensor): Mask of value sequence. 
            0s for padding, otherwise 1s. [bsz, Lv]
        """
        bsz, Lq, Lv = *query.size()[:2], value.size(1)

        # 1. project query
        # [bsz, Lq, v_size]
        projected_query = self.query_proj(query)

        # 2. compute attention score by dot-prodcut 
        # vanilla dot-product
        # q = projected_query[:, :, None, :]
        # v = value[:, None, :, :]
        # e = (q * v).sum(-1)
        
        # vecorized dot-product 
        # [bsz, Lq, v_size] \dot [bsz, Lv, v_size]^T -> [bsz, Lq, Lv]
        # using torch.bmm (batch matrix multiplication)
        q, v = projected_query, value
        e = torch.bmm(q, v.transpose(-1, -2))  # or q @ v.transpose(-1, -2)

        # assign -inf to the scores of paddings so that after softmax
        # they will be 0.
        padded_mask = (1-value_mask)[:, None, :].repeat(1, Lq, 1)
        e.masked_fill_(padded_mask, -float("inf"))

        # 3. compute attention scores using softmax over value
        # [bsz, Lq, Lv]
        scores = F.softmax(e, dim=-1)

        # 4. compute attention output by weighted average over 
        #   values with attention scores
        # vanilla weighted average
        # outputs = (scores[:, :, :, None] * value[:, None, :, :]).sum(2)

        # vectorized weighted average
        # [bsz, Lq, Lv] \dot [bsz, Lv, v_size] -> [bsz, Lq, v_size]
        outputs = torch.bmm(scores, value)

        return outputs, scores

