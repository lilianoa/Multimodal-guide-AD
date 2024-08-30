# stacked cross-attention fusion

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SAF(nn.Module):

    def __init__(self,
            input_dims,
            output_dim,
            mm_dim=512,
            num_atten_layer=2,
            multi_head=8,
            dropout=0.2):
        super(SAF, self).__init__()
        self.input_dims = input_dims
        self.hidden_size = mm_dim
        self.output_dim = output_dim
        self.atten_layer = num_atten_layer
        self.multi_head = multi_head

        # Modules
        self.san = nn.ModuleList([Attention(self.input_dims, self.hidden_size, self.output_dim, self.multi_head, dropout)] * self.atten_layer)
        self.norm = nn.BatchNorm1d(self.output_dim)
        self.n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def forward(self, xi, xt):

        v = xt
        q = xi
        k = xt
        '''
        v = xi
        q = xt
        k = xi
        '''
        for att_layer in self.san:
            attn = att_layer(v, q, k)
            u = torch.add(F.relu(self.norm(attn)), v)  # Note: Do not use += , use torch.add() instead.
            v = u
            k = u
        return u


class Attention(nn.Module):
    def __init__(self, input_dims, hidden_size, output_dim, multi_head, dropout):
        super(Attention, self).__init__()

        self.multi_head = multi_head
        self.hidden_size = hidden_size

        self.linear_v = nn.Linear(input_dims[1], hidden_size)
        self.linear_k = nn.Linear(input_dims[1], hidden_size)
        self.linear_q = nn.Linear(input_dims[0], hidden_size)
        self.linear_merge = nn.Linear(hidden_size, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, v, k, q):
        n_batches = q.size(0)
        v = self.linear_v(v).view(
            n_batches,
            -1,
            self.multi_head,
            int(self.hidden_size / self.multi_head)
        ).transpose(1, 2)
        k = self.linear_k(k).view(
            n_batches,
            -1,
            self.multi_head,
            int(self.hidden_size / self.multi_head)
        ).transpose(1, 2)

        q = self.linear_q(q).view(
            n_batches,
            -1,
            self.multi_head,
            int(self.hidden_size / self.multi_head)
        ).transpose(1, 2)

        atted = self.att(v, k, q)
        atted = atted.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
            self.hidden_size
        )

        atted = self.linear_merge(atted).squeeze()

        return atted

    def att(self, value, key, query):
        d_k = query.size(-1)

        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)

        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)

        return torch.matmul(att_map, value)
