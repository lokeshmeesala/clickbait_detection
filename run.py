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


click_bait_data = pd.read_csv('data/click_bait_full.csv')
n = int(0.9*len(click_bait_data)) # first 90% will be train, rest val
train_data = click_bait_data[:n]
val_data = click_bait_data[n:]



datasets_train_test = DatasetDict({
    "train": Dataset.from_pandas(train_data),
    "val": Dataset.from_pandas(val_data)
    })


checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint, return_tensors='pt')


def tokenize_function(example):
    return tokenizer(example["headline"], truncation=True)


tokenized_datasets = datasets_train_test.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors='pt')

train_dataset = tokenized_datasets['train'].remove_columns(['id', 'headline', 'body', 'truthMean',])
val_dataset = tokenized_datasets['val'].remove_columns(['id', 'headline', 'body', 'truthMean',])

train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=16, collate_fn=data_collator)
val_dataloader = DataLoader(val_dataset, shuffle=True, batch_size=16, collate_fn=data_collator)

# hyperparameters
batch_size = 16 # how many independent sequences will we process in parallel?
block_size = 512 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 512
n_head = 4
n_layer = 4
dropout = 0.2
vocab_size = tokenizer.vocab_size
output_dim = 2

# # data loading
# def get_batch(split):
#     # generate a small batch of data of inputs x and targets y
#     data = train_dataloader if split == 'train' else val_dataloader
#     # ix = torch.randint(len(data) - block_size, (batch_size,))
#     # x = torch.stack([data[i:i+block_size] for i in ix])
#     # y = torch.stack([data[i+1:i+block_size+1] for i in ix])
#     x, y = x.to(device), y.to(device)
#     return x, y



class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
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

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.classification_head = nn.Linear(vocab_size, output_dim)
        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # print(idx)
        B, T = idx.shape
        print(idx.shape)
        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        x = self.lm_head(x) # (B,T,vocab_size)
        pooled_output = torch.mean(x, dim=1)
        # print("1",pooled_output.shape,pooled_output)
        logits = self.classification_head(pooled_output)
        # print("2",logits.shape,logits)
        # B, T, C = x.shape
        # x = x.view(B*T, C)
        # print("3",x.shape,x)
        # logits = nn.Linear(B*T, C)(x)

        if targets is None:
            loss = None
        else:
            print("a",logits.shape)
            # print(targets)
            print("b",targets.shape)
            loss = F.cross_entropy(logits, targets)

        return logits, loss


model = GPTLanguageModel()
m = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# @torch.no_grad()
# def estimate_loss(tr_xb, tr_yb, val_xb, val_yb):
#     out = {}
#     model.eval()
#     for split in ['train', 'val']:
#         losses = torch.zeros(eval_iters)
#         for k in range(eval_iters):
#             # X, Y = get_batch(split)
#             logits, loss = model(xb, yb)
#             losses[k] = loss.item()
#         out[split] = losses.mean()
#     model.train()
#     return out

def get_metric(pred, act):
    pred  = torch.argmax(pred,dim=1)
    # print(pred)
    # print(act)
    print("accuracy", sum(pred == act) / len(pred))
    
for step, batch in enumerate(train_dataloader):
    # print(batch.keys())
    xb, yb = batch['input_ids'], batch['labels']
    logits, loss = model(xb, yb)
    # print(logits)
    print("loss",loss)
    get_metric(logits, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    if step == 15:
        print("pred", torch.argmax(logits,dim=1))
        print("actual", yb)