import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from copy import deepcopy


def print_memory_usage(tag=""):
    allocated = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    print(f"{tag} - Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB")

def _get_clones(module, N):
    return nn.ModuleList([deepcopy(module) for i in range(N)])

class MultiHeadedCrossAttn(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.3):
        super(MultiHeadedCrossAttn, self).__init__()
        self.num_heads = num_heads
        self.embed_dim = dim
        assert dim % num_heads == 0
        self.head_dim = dim // num_heads
        self.q_proj = nn.Linear(dim,dim)
        self.k_proj = nn.Linear(dim,dim)
        self.v_proj = nn.Linear(dim,dim)
        self.out_proj = nn.Linear(dim,dim)
#
    def forward(self, q_embeds, context_embeds, attn_mask=None):

        t, b, a, d = q_embeds.shape # actually, t=1
        q = self.q_proj(q_embeds)
        q = q.reshape(t, b, a, self.num_heads, self.head_dim) # (t,b,a,h,hd)
        q = q.permute(0,1,3,4,2) # (t,b,h,hd,a)

        _, f, _, _ = context_embeds.shape
        video_embeds = context_embeds.permute(0,2,1,3) # (t,b,f,d)
        k = self.k_proj(context_embeds)
        k = k.reshape(t, b, f, self.num_heads, self.head_dim)
        k = k.permute(0,1,3,2,4) # (t,b,h,f,hd)

        v = self.v_proj(video_embeds)
        v = v.reshape(t, b, f, self.num_heads, self.head_dim)
        v = v.permute(0,1,3,4,2) # (t,b,h,hd,f)

        # (t,b,h,f,hd)x(t,b,h,hd,a)->(t,b,h,f,a)
        # this is pair-wise parallelization
        attention_logits = k @ q
        attention_logits = attention_logits / math.sqrt(self.head_dim)
        if attn_mask is not None:
            # attn_mask shape: (b,f), valid for 1, invalid for 0
            attn_mask = attn_mask[None,:,None,:,None] # ->(1,b,1,f,1)
            attention_logits = attention_logits.masked_fill(attn_mask == 0, -1e9)
        attention_weights = F.softmax(attention_logits, dim=-2)

        attention = v @ attention_weights
        # (t,b,h,hd,f)x(t,b,h,f,a)->(t,b,h,hd,a)

        attention = attention.permute(0,1,4,2,3)  # (t,b,a,h,hd)
        attention = attention.reshape(t, b, a, self.embed_dim)

        o = self.out_proj(attention)
        return o.permute(0,2,1,3), attention_weights # (a,b,d)


class Transformer(nn.Module):
    def __init__(self, embed_dim, attn_heads, dropout=0.3):
        super(Transformer, self).__init__()
        self.embed_dim = embed_dim

        self.cross_attn = MultiHeadedCrossAttn(embed_dim, attn_heads, dropout=dropout)

        self.linear1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.linear2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.relu = nn.ReLU(inplace=True)
        self.mlp = nn.Sequential(
            nn.Linear(self.embed_dim, 4*embed_dim), nn.ReLU(inplace=True),
            nn.Linear(4*self.embed_dim, self.embed_dim)
        )

        self.layer_norm1 = nn.LayerNorm(self.embed_dim)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim)
        self.layer_norm3 = nn.LayerNorm(self.embed_dim)
        self.dropout = nn.Dropout(dropout)

        self._init_parameters()

    def _init_parameters(self):
        for name, param in self.named_parameters():
            if 'linear' in name or 'proj' in name:
                if 'weight' in name:
                    nn.init.eye_(param)
                elif 'bias' in name:
                    param.data.fill_(0.)

    def forward(self, q_embeds, C_embeds):
        # pre norm
        q_embeds = self.layer_norm1(q_embeds)
        C_embeds = self.layer_norm1(C_embeds) # it's important to use the same layer norm

        attn_out,_ = self.cross_attn(q_embeds, C_embeds) # (a,b,d)
        attn_out = self.layer_norm2(attn_out)

        linear_out = self.linear1(attn_out)
        out = attn_out + self.dropout(linear_out)
        out = self.layer_norm3(out)

        return out


class Psi(nn.Module):
    def __init__(self, dim, num_mha_heads, num_layers=1, dropout=0.3):
        super(Psi, self).__init__()
        cross_layer = Transformer(dim, num_mha_heads, dropout)
        self.num_layers = num_layers
        self.layers = _get_clones(cross_layer, num_layers)
        # self.cross_attn = Transformer(dim, num_mha_heads, dropout)

        self.mean_proj = nn.Linear(dim, dim)
        self.log_std_proj = nn.Linear(dim, dim)

    def forward(self, delta, C):
        t,a,b,d = delta.size()
        delta = delta.permute(0,2,1,3) # (t,b,a,d)
        C = C.permute(0,2,1,3)

        out = delta
        for layer in self.layers:
            _out = layer(out, C) # ->(t,a,b,d)
        mean = self.mean_proj(_out) # (t,a,b,d)
        log_sigma = self.log_std_proj(_out) # (t,a,b,d)
        return mean, log_sigma



class Agent(nn.Module):
    def __init__(self, dim=512, num_heads=8, num_layers=1, dropout=0.3):
        super(Agent, self).__init__()

        self.psi = Psi(dim, 1)
