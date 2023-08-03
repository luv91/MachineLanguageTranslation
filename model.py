import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import dataset
import numpy as np
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import json
import pickle
# import matplotlib.pyplot as plt


class MultiHeadAttention(nn.Module):
    
    def __init__(self, d_k, d_model, n_heads, max_len, causal = False):
        
        super().__init__()
        
        # assume d_v = d_k
        self.d_k = d_k
        self.n_heads = n_heads
        
        self.key = nn.Linear(d_model, d_k*n_heads)
        self.query = nn.Linear(d_model, d_k*n_heads)
        self.value = nn.Linear(d_model, d_k*n_heads)
        
        # final linear layer
        self.fc = nn.Linear(d_k* n_heads, d_model)
        
        # causal mask
        # make it so that diagonal is 0 too.. 
        # this way we dont have to shift the inputs to make targets
        
        self.causal = causal
        
        # causal mask if causal has been set to true. 
        if causal:
            
            cm = torch.tril(torch.ones(max_len, max_len))
            
            self.register_buffer(
                "causal_mask",
                cm.view(1,1, max_len, max_len)
            )
            
    def forward(self, q,k,v, pad_mask = None):
        
        q = self.query(q)  # N X T X (hd_k)
        k = self.key(k)  # N X T X (hd_k)
        v = self.value(v)  # N X T X (hd_k)
        
        # in seq-2-seq, it is possible to apply attention where we want to know, 
        # which decoder output should pay attention to which encoder input. 
        # this is a cross-attention part, wehre encoder is connected to the decoder. 
        # k and v comes from the encoder while q (query ) comes from the decoder. 
        # so input to this layer comes from encoder input (k), while outptu shape will be q (since output comes from query in decoder)
        N = q.shape[0]
        T_output = q.shape[1]
        T_input = k.shape[1]
        
        #change the shape to:
        # (N,T,h,d_k) --> (N,h,T,d_k)
        
        # in order for amtrix multiply to work properly
        
        q = q.view(N, T_output, self.n_heads, self.d_k).transpose(1,2)
        k = k.view(N, T_input, self.n_heads, self.d_k).transpose(1,2)
        v = v.view(N, T_input, self.n_heads, self.d_k).transpose(1,2)
        
        # compute attention weights
        # (N,h,T, d_k) x(N,h,d_k, T) --> (N,h,T,T)
        
        attn_scores = q @ k.transpose(-2,-1) / math.sqrt(self.d_k)
        
        if pad_mask is not None:
            
            attn_scores = attn_scores.masked_fill(
                pad_mask[:,None,None,:] == 0, float('-inf'))
            
        # what changes is how we apply thhe causal mask.. 
        if self.causal:
            attn_scores = attn_scores.masked_fill(
                self.causal_mask[:,:,:T_output, :T_input] == 0, float('-inf'))
            
        attn_weights = F.softmax(attn_scores, dim = -1)
        
        
        # compute attention-weighted values
        # (N,h,T,T) x (N,h,T,d_k) --> (N,h, T, d_k)
        A = attn_weights @ v
        
        # reshape it back before final linear layer
        A = A.transpose(1,2)  # (N,T,h,d_k)
        
        A = A.contiguous().view(N,T_output, self.d_k*self.n_heads)   # (N, T, h*d_k)
        
        # projection 
        return self.fc(A)
    
class EncoderBlock(nn.Module):
    
    def __init__(self, d_k, d_model, n_heads, max_len,dropout_prob = 0.1):
        
        super().__init__()
        
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.mha = MultiHeadAttention(d_k, d_model, n_heads, max_len, causal = False)
        self.ann = nn.Sequential(
            nn.Linear(d_model, d_model*4),
            nn.GELU(), 
            nn.Linear(d_model*4, d_model),
            nn.Dropout(dropout_prob),
        )
        
        self.dropout = nn.Dropout(p = dropout_prob)
        
        
    def forward(self, x, pad_mask = None):
        
        x = self.ln1(x+self.mha(x,x,x,pad_mask))
        x = self.ln2(x+ self.ann(x))
        x = self.dropout(x)
        return x
    

class DecoderBlock(nn.Module):
    
    def __init__(self, d_k, d_model, n_heads, max_len,dropout_prob = 0.1):
        
        super().__init__()
        
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ln3 = nn.LayerNorm(d_model)
        self.mha1 = MultiHeadAttention(d_k, d_model, n_heads, max_len, causal = True)  # this layer is cross-attention layer
        self.mha2 = MultiHeadAttention(d_k, d_model, n_heads, max_len, causal = False)
        self.ann = nn.Sequential(
            nn.Linear(d_model, d_model*4),
            nn.GELU(), 
            nn.Linear(d_model*4, d_model),
            nn.Dropout(dropout_prob),
        )
        
        self.dropout = nn.Dropout(p = dropout_prob)
        
        
    def forward(self, enc_output, dec_input, enc_mask = None, dec_mask = None):
        
        x = self.ln1(
            dec_input + self.mha1(dec_input, dec_input, dec_input, dec_mask)
        )
        
        # multi-head attention including encoder output
        
        x = self.ln2(x + self.mha2(x, enc_output, enc_output, enc_mask))
        
        x = self.ln3(x+self.ann(x))
        
        x = self.dropout(x)
        
        return x
    
class PositionalEncoding(nn.Module):
    
    def __init__(self, d_model, max_len = 2048, dropout_prob = 0.1):
        
        super().__init__()
        
        self.dropout = nn.Dropout(p = dropout_prob)
        
        position = torch.arange(max_len).unsqueeze(1)
        exp_term = torch.arange(0, d_model, 2)
        div_term = torch.exp(exp_term* (-math.log(10000.0)/ d_model))

        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position* div_term)
        pe[0,:, 1::2] = torch.cos(position* div_term)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        
        # x.shape: N xT X D
        x = x + self.pe[:,:x.size(1), :]
        return self.dropout(x)
    
class Encoder(nn.Module):
    
    def __init__(self, vocab_size, max_len, d_k, d_model, 
                 n_heads, n_layers, dropout_prob):
        
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout_prob)
        
        transformer_blocks = [
            
            EncoderBlock(d_k, 
                         d_model, 
                         n_heads, 
                         max_len,
                         dropout_prob) for _ in range(n_layers)]
        
        self.transformer_blocks = nn.Sequential(*transformer_blocks)
        self.ln = nn.LayerNorm(d_model)
        
    def forward(self, x, pad_mask = None):
        
        x = self.embedding(x)
        x = self.pos_encoding(x)
        
        for block in self.transformer_blocks:
            
            x = block(x, pad_mask)
            
        x = self.ln(x)
            
        return x
        

class Decoder(nn.Module):
    
    def __init__(self, vocab_size, max_len, d_k, d_model, 
                 n_heads, n_layers, dropout_prob):
        
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout_prob)
        
        transformer_blocks = [
            
            DecoderBlock(d_k, 
                         d_model, 
                         n_heads, 
                         max_len,
                         dropout_prob) for _ in range(n_layers)]
        
        self.transformer_blocks = nn.Sequential(*transformer_blocks)
        self.ln = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, vocab_size)
        
    def forward(self, enc_output, dec_input, enc_mask = None, dec_mask = None):
        
        x = self.embedding(dec_input)
        x = self.pos_encoding(x)
        
        for block in self.transformer_blocks:
            x = block(enc_output, x, enc_mask, dec_mask)
            
        x = self.ln(x)
        x = self.fc(x)  # many - to -many

        return x
            
class Transformer(nn.Module):
    
    def __init__(self, encoder, decoder):
        
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        
        
    def forward(self,enc_input, dec_input, enc_mask, dec_mask):
        
        enc_output = self.encoder(enc_input, enc_mask)
        dec_output = self.decoder(enc_output, dec_input, enc_mask, dec_mask)
        
        return dec_output