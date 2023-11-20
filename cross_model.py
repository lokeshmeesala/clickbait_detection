import pandas as pd
import numpy as np
import math
import json

import torch
import torch.nn as nn
from torch.nn import functional as F
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer, DataCollatorWithPadding
from torch.utils.data.dataloader import DataLoader
from params import *
from utils import *

# from run import vocab_size

# hyperparameters
# batch_size = 32 # how many independent sequences will we process in parallel?
# block_size = 1024 # what is the maximum context length for predictions?
# learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# n_embd = 256
# n_head = 4
# n_layer = 4
# dropout = 0.2
# output_dim = 2

torch.manual_seed(1337)
print(f"DEVICE {device}")

# data loading
def get_batch(dataloader):
    for _, batch in enumerate(dataloader):
        x, y = batch['input_ids'], batch['labels']
        break
    x, y = x.to(device), y.to(device)
    return x, y


class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size, i):
        super().__init__()
        self.i = i
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.register_buffer('tril', torch.ones(block_size, block_size))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x_head, x_body):
        #print("\t\t\tin head", self.i)
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        # B,T,C = x_head.shape
        k_head = self.key(x_head)   # (B,T,hs)
        #print("\t\t\tk_head.shape",k_head.shape)
        # q_body = self.query(x_body) # (B,T,hs) <----- CROSS ATTENTION
        q_head = self.query(x_head) # (B,T,hs) <----- SELF ATTENTION
        #print("\t\t\tq_body.shape",q_body.shape)
        # compute attention scores ("affinities")
        wei_head = q_head @ k_head.transpose(-2,-1) * k_head.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        #print("wei_head.shape",wei_head.shape)
        wei_head = wei_head.masked_fill(self.tril[:x_head.shape[1], :x_head.shape[1]] == 0, float('-inf')) # (B, T, T)
        wei_head = F.softmax(wei_head, dim=-1) # (B, T, T)
        wei_head = self.dropout(wei_head)
        # perform the weighted aggregation of the values
        v_head = self.value(x_head) # (B,T,hs)
        #print("\t\t\tv.shape",v_head.shape)
        out_head = wei_head @ v_head # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        
        
        k_body = self.key(x_body)   # (B,T,hs)
        #print("\t\t\tk_body.shape",k_body.shape)
        q_body = self.query(x_body) # (B,T,hs) <----- SELF ATTENTION
        # q_head = self.query(x_head) # (B,T,hs) <-----  CROSS ATTENTION
        #print("\t\t\tq_head.shape",q_head.shape)
        # compute attention scores ("affinities")
        wei_body = q_body @ k_body.transpose(-2,-1) * k_body.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        #print("wei_body.shape",wei_body.shape)
        wei_body = wei_body.masked_fill(self.tril[:x_body.shape[1], :x_body.shape[1]] == 0, float('-inf')) # (B, T, T)
        wei_body = F.softmax(wei_body, dim=-1) # (B, T, T)
        wei_body = self.dropout(wei_body)
        # perform the weighted aggregation of the values
        v_body = self.value(x_body) # (B,T,hs)
        #print("\t\t\tv.shape",v_body.shape)
        out_body = wei_body @ v_body # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        
        
        #print("\t\t\out_head.shape",out_head.shape)
        #print("\t\t\out_body.shape",out_body.shape)
        # out = ((out_head+out_body)/2)
        return out_head, out_body

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, i) for i in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_head, x_body):
        #print("\t\tin sa")
        # #print("\t\tx.shape", x.shape)
        all_outputs  = [h(x_head, x_body) for h in self.heads]
        out_head = torch.cat([i[0] for i in all_outputs], dim=-1)
        out_body = torch.cat([i[1] for i in all_outputs], dim=-1)    
        # out_head, out_body = torch.cat([h(x_head, x_body) for h in self.heads], dim=-1)
        #print("\t\tafter heads")
        #print("\t\tout_head.shape", out_head.shape)
        #print("\t\tout_body.shape", out_body.shape)
        out_head = self.dropout(self.proj(out_head))
        out_body = self.dropout(self.proj(out_body))
        return out_head, out_body

class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)
    
class ClassificationHead(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(16, 2),
            nn.Sigmoid()
        )

    def forward(self, x):
        output = self.net(x)
        return output
    
class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head, i):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.i = i
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, full_x):
        x_head, x_body = full_x[0], full_x[1]
        #print("in block", self.i)
        # #print("\tx before sa", x.shape)
        # x = x + self.ln1(self.sa(x_head, x_body))  ## ADD RES CONNECTION
        x_head, x_body = self.sa(x_head, x_body)
        
        x_head = x_head + self.ln1(x_head)
        x_head = x_head + self.ln2(self.ffwd(x_head))

        x_body = x_body + self.ln1(x_body)
        x_body = x_body + self.ln2(self.ffwd(x_body))

        # #print("\tx after sa", x.shape)
        return [x_head, x_body]

class CrossEncoder(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head, i=i) for i in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        # self.lm_head = nn.Linear(n_embd, vocab_size)
        # self.classification_head = nn.Linear(vocab_size, output_dim)
        # self.classification_head = nn.Linear(n_embd, output_dim)
        self.classification_head = ClassificationHead(n_embd)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx_head, idx_body, targets=None):
        # print(idx)
        # B, T = x_head.shape
        #print("idx_head.shape", idx_head.shape)
        #print("idx_body.shape", idx_body.shape)
        # #print("min",idx[0].min())
        # #print("max",idx[0].max())
        # idx and targets are both (B,T) tensor of integers
        tok_emb_head = self.token_embedding_table(idx_head) # (B,T,C)
        tok_emb_body = self.token_embedding_table(idx_body) # (B,T,C)

        #print("tok_emb_head.shape", tok_emb_head.shape)
        #print("tok_emb_body.shape", tok_emb_body.shape)
        # #print("TOKEN EMBED CHECKED")
        pos_emb_head = self.position_embedding_table(torch.arange(idx_head.shape[1], device=device)) # (T,C)
        pos_emb_body = self.position_embedding_table(torch.arange(idx_body.shape[1], device=device)) # (T,C)

        #print("pos_emb_head.shape", pos_emb_head.shape)
        #print("pos_emb_body.shape", pos_emb_body.shape)

        x_head = tok_emb_head + pos_emb_head # (B,T,C)
        x_body = tok_emb_body + pos_emb_body # (B,T,C)
        #print("before blocks x_head.shape", x_head.shape)
        #print("before blocks x_body.shape", x_body.shape)
        full_x = [x_head, x_body]
        full_x = self.blocks(full_x) # (B,T,C)
        
        x_head, x_body = full_x[0], full_x[1]
        #print("after blocks x_head.shape", x_head.shape)
        #print("before ln_f x_head.shape", x_head.shape)
        #print("after blocks x_body.shape", x_body.shape)
        #print("before ln_f x_body.shape", x_body.shape)
        # x_head = self.ln_f(x_head) # (B,T,C)
        # x_body = self.ln_f(x_body) # (B,T,C)
        # x = ((x_head+x_body)/2)
        x = self.ln_f(x_head)
        #print("after ln_f x.shape", x.shape)
        #print("before lm_head x.shape", x.shape)
        # x = self.lm_head(x) # (B,T,vocab_size)
        #print("after lm_head x.shape", x.shape)
        #print("before pooled x.shape", x.shape)
        pooled_output = torch.mean(x, dim=1)
        #print("after pooled pooled_output.shape", pooled_output.shape)
        # #print("1",pooled_output.shape,pooled_output)
        #print("before classification_head pooled_output.shape", pooled_output.shape)
        logits = self.classification_head(pooled_output)
        #print("after classification_head logits.shape", logits.shape)
        return logits