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
device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(1337)
print(f"DEVICE {device}")

class SA_Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size, i):
        super().__init__()
        self.i = i
        self.key_head = nn.Linear(n_embd, head_size, bias=False)
        self.query_head = nn.Linear(n_embd, head_size, bias=False)
        self.value_head = nn.Linear(n_embd, head_size, bias=False)
        
        self.key_body = nn.Linear(n_embd, head_size, bias=False)
        self.query_body = nn.Linear(n_embd, head_size, bias=False)
        self.value_body = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_head, x_body):
        k_head = self.key_head(x_head)
        q_head = self.query_head(x_head)
        v_head = self.value_head(x_head)
                
        k_body = self.key_body(x_body)
        q_body = self.query_body(x_body)
        v_body = self.value_body(x_body)

        # HEAD - SELF ATTENTION
        affinity_head = q_head @ k_head.transpose(-2,-1) * k_head.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        affinity_head = F.softmax(affinity_head, dim=-1) # (B, T, T)
        affinity_head = self.dropout(affinity_head) # (B,T,hs)
        out_head = affinity_head @ v_head 
        
        # BODY - SELF ATTENTION
        affinity_body = q_body @ k_body.transpose(-2,-1) * k_body.shape[-1]**-0.5 #B T 16 @ B, 16, T
        affinity_body = F.softmax(affinity_body, dim=-1)
        affinity_body = self.dropout(affinity_body)
        out_body = affinity_body @ v_body
        return (out_head,out_body)
  
class CA_Head(nn.Module):
    def __init__(self, head_size, i):
        super().__init__()
        self.i = i
        self.key_head = nn.Linear(n_embd, head_size, bias=False)
        self.query_head = nn.Linear(n_embd, head_size, bias=False)
        self.value_head = nn.Linear(n_embd, head_size, bias=False)
        
        self.key_body = nn.Linear(n_embd, head_size, bias=False)
        self.query_body = nn.Linear(n_embd, head_size, bias=False)
        self.value_body = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_head, x_body):
        k_head = self.key_head(x_head)
        q_head = self.query_head(x_head)
        v_head = self.value_head(x_head)
                
        k_body = self.key_body(x_body)
        q_body = self.query_body(x_body)
        v_body = self.value_body(x_body)
        
        ## HEAD - CROSS ATTENTION
        affinity_head = q_head @ k_body.transpose(-2,-1) * k_body.shape[-1]**-0.5 #B T 16 @ B, 16, T
        affinity_head = F.softmax(affinity_head, dim=-1)
        affinity_head = self.dropout(affinity_head)
        out_head = affinity_head @ v_body
        
        ## BODY - CROSS ATTENTION
        affinity_body = q_body @ k_head.transpose(-2,-1) * k_head.shape[-1]**-0.5 #B T 16 @ B, 16, T
        affinity_body = F.softmax(affinity_body, dim=-1)
        affinity_body = self.dropout(affinity_body)
        out_body = affinity_body @ v_head
        
        return (out_head,out_body)
    
class MultiHeadSAAttention(nn.Module):

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.sa_heads = nn.ModuleList([SA_Head(head_size, i) for i in range(num_heads)])
        self.proj_head = nn.Linear(head_size * num_heads, n_embd)
        self.proj_body = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_head, x_body):
        all_outputs  = [h(x_head, x_body) for h in self.sa_heads]
        out_head = torch.cat([i[0] for i in all_outputs], dim=-1)
        out_body = torch.cat([i[1] for i in all_outputs], dim=-1)
        out_head = self.dropout(self.proj_head(out_head))
        out_body = self.dropout(self.proj_body(out_body))
        return (out_head,out_body)

class MultiHeadCAAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.ca_heads = nn.ModuleList([CA_Head(head_size, i) for i in range(num_heads)])
        self.proj_head = nn.Linear(head_size * num_heads, n_embd)
        self.proj_body = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_head, x_body):
        all_outputs  = [h(x_head, x_body) for h in self.ca_heads]
        out_head = torch.cat([i[0] for i in all_outputs], dim=-1)
        out_body = torch.cat([i[1] for i in all_outputs], dim=-1)
        out_head = self.dropout(self.proj_head(out_head))
        out_body = self.dropout(self.proj_body(out_body))
        return (out_head,out_body)
    
class Block(nn.Module):
    def __init__(self, n_embd, n_head, i):
        super().__init__()
        head_size = n_embd // n_head
        self.i = i
        self.sa = MultiHeadSAAttention(n_head, head_size)
        self.ca = MultiHeadCAAttention(n_head, head_size)
        self.ffwd_head = FeedForward(n_embd)
        self.ffwd_body = FeedForward(n_embd)
        self.ln1_head = nn.LayerNorm(n_embd)
        self.ln1_body = nn.LayerNorm(n_embd)
        self.ln2_head = nn.LayerNorm(n_embd)
        self.ln2_body = nn.LayerNorm(n_embd)

    def forward(self, x_full):
        x_head = x_full[0]
        x_body = x_full[1]
        
        sa_x_head, sa_x_body = self.sa(x_head,x_body)
        x_head = x_head + self.ln1_head(sa_x_head)
        x_head = x_head + self.ln2_head(self.ffwd_head(x_head))
        x_body = x_body + self.ln1_body(sa_x_body)
        x_body = x_body + self.ln2_body(self.ffwd_body(x_body))
        
        ca_x_head, ca_x_body = self.ca(x_head,x_body)
        x_head = x_head + self.ln1_head(ca_x_head)
        x_head = x_head + self.ln2_head(self.ffwd_head(x_head))
        x_body = x_body + self.ln1_body(ca_x_body)
        x_body = x_body + self.ln2_body(self.ffwd_body(x_body))

        x_full = [x_head, x_body]
        return x_full

class FeedForward(nn.Module):
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
            nn.Linear(2*input_size, 4*input_size),
            nn.ReLU(),
            nn.Linear(4*input_size, input_size),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(input_size, input_size//2),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(input_size//2, input_size//4),
            nn.ReLU(),
            nn.Linear(input_size//4, 2),
            nn.Sigmoid()
        )

    def forward(self, x):
        output = self.net(x)
        return output

class CrossEncoder(nn.Module):

    def __init__(self, vocab_size, custom_embeddings=None):
        super().__init__()
        self.use_custom_embeddings = False
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        if custom_embeddings != None:
            self.use_custom_embeddings = True
            self.custom_embedding_table = nn.Embedding.from_pretrained(custom_embeddings, freeze=True)
            
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head, i=i) for i in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.classification_head = ClassificationHead(n_embd)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx_head, idx_body):
        tok_emb_head = self.token_embedding_table(idx_head)
        tok_emb_body = self.token_embedding_table(idx_body)

        pos_emb_head = self.position_embedding_table(torch.arange(idx_head.shape[1], device=device))
        pos_emb_body = self.position_embedding_table(torch.arange(idx_body.shape[1], device=device))

        if self.use_custom_embeddings:
            cus_emb_head = self.custom_embedding_table(idx_head)
            cus_emb_body = self.custom_embedding_table(idx_body)
            x_head = tok_emb_head + cus_emb_head + pos_emb_head
            x_body = tok_emb_body + cus_emb_body + pos_emb_body
        
        else:
            x_head = tok_emb_head + pos_emb_head
            x_body = tok_emb_body + pos_emb_body
        

        x_full = [x_head, x_body]
        x_full = self.blocks(x_full)
        x_head = x_full[0]
        x_body = x_full[1]
        
        x_head = self.ln_f(x_head)
        x_body = self.ln_f(x_body)
        
        pooled_output  = torch.cat([x_head.mean(dim=1), x_body.mean(dim=1)], dim=1)
        
        logits = self.classification_head(pooled_output)
        return logits