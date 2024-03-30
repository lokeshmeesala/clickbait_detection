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
from src.utils import *

params = load_yaml('./params.yaml')

n_embd = params['n_embd']
dropout = params['dropout']
block_size = params['block_size']
n_head = params['n_head']
n_layer = params['n_layer']



device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(1337)
print(f"DEVICE {device}")

class SA_Head(nn.Module):
    """ 
    One head of self-attention 
    """
    def __init__(self, head_size, i):
        super().__init__()
        self.i = i

        # Initialize linear layers for the head and body of self-attention
        self.key_head = nn.Linear(n_embd, head_size, bias=False)
        self.query_head = nn.Linear(n_embd, head_size, bias=False)
        self.value_head = nn.Linear(n_embd, head_size, bias=False)
        
        self.key_body = nn.Linear(n_embd, head_size, bias=False)
        self.query_body = nn.Linear(n_embd, head_size, bias=False)
        self.value_body = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_head, x_body):
        """
        Compute key, query, and value for the head and body
        """
        k_head = self.key_head(x_head)
        q_head = self.query_head(x_head)
        v_head = self.value_head(x_head)
                
        k_body = self.key_body(x_body)
        q_body = self.query_body(x_body)
        v_body = self.value_body(x_body)

        # HEAD - SELF ATTENTION
        # Compute self attention scores between a head and itself, apply softmax, and apply dropout
        affinity_head = q_head @ k_head.transpose(-2,-1) * k_head.shape[-1]**-0.5
        affinity_head = F.softmax(affinity_head, dim=-1)
        affinity_head = self.dropout(affinity_head)
        out_head = affinity_head @ v_head 
        
        # BODY - SELF ATTENTION
        # Compute self attention scores between a body and itself, apply softmax, and apply dropout
        affinity_body = q_body @ k_body.transpose(-2,-1) * k_body.shape[-1]**-0.5
        affinity_body = F.softmax(affinity_body, dim=-1)
        affinity_body = self.dropout(affinity_body)
        out_body = affinity_body @ v_body

        # Return self aware head and body vectors
        return (out_head,out_body)
  
class CA_Head(nn.Module):
    """One head of cross-attention."""
    def __init__(self, head_size, i):
        super().__init__()
        # Initialize linear layers for the head and body of cross-attention
        
        self.i = i
        self.key_head = nn.Linear(n_embd, head_size, bias=False)
        self.query_head = nn.Linear(n_embd, head_size, bias=False)
        self.value_head = nn.Linear(n_embd, head_size, bias=False)
        
        self.key_body = nn.Linear(n_embd, head_size, bias=False)
        self.query_body = nn.Linear(n_embd, head_size, bias=False)
        self.value_body = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_head, x_body):
        """Compute key, query, and value for the head and body."""
        k_head = self.key_head(x_head)
        q_head = self.query_head(x_head)
        v_head = self.value_head(x_head)
                
        k_body = self.key_body(x_body)
        q_body = self.query_body(x_body)
        v_body = self.value_body(x_body)
        
        ## HEAD - CROSS ATTENTION
        # Compute attention scores between head query and body key, apply softmax, and apply dropout
        affinity_head = q_head @ k_body.transpose(-2,-1) * k_body.shape[-1]**-0.5
        affinity_head = F.softmax(affinity_head, dim=-1)
        affinity_head = self.dropout(affinity_head)
        out_head = affinity_head @ v_body
        
        ## BODY - CROSS ATTENTION
        # Compute attention scores between body query and head key, apply softmax, and apply dropout
        affinity_body = q_body @ k_head.transpose(-2,-1) * k_head.shape[-1]**-0.5
        affinity_body = F.softmax(affinity_body, dim=-1)
        affinity_body = self.dropout(affinity_body)
        out_body = affinity_body @ v_head
        
        # Return head and body vectors that have information about each other.
        return (out_head,out_body)
    
class MultiHeadSAAttention(nn.Module):
    """Multi-head self-attention layer."""
    def __init__(self, num_heads, head_size):
        super().__init__()
        # Initialize self-attention heads
        self.sa_heads = nn.ModuleList([SA_Head(head_size, i) for i in range(num_heads)])

        # Projection layers for the head and body outputs
        self.proj_head = nn.Linear(head_size * num_heads, n_embd)
        self.proj_body = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_head, x_body):
        # Process inputs through all self-attention heads
        all_outputs  = [h(x_head, x_body) for h in self.sa_heads]

        # Concatenate outputs from all self-attention heads along the last dimension
        out_head = torch.cat([i[0] for i in all_outputs], dim=-1)
        out_body = torch.cat([i[1] for i in all_outputs], dim=-1)

        # Apply dropout and project the concatenated outputs back to the original embedding size
        out_head = self.dropout(self.proj_head(out_head))
        out_body = self.dropout(self.proj_body(out_body))
        return (out_head,out_body)

class MultiHeadCAAttention(nn.Module):
    """Multi-head cross-attention layer."""

    def __init__(self, num_heads, head_size):
        super().__init__()
        # Initialize cross-attention heads
        self.ca_heads = nn.ModuleList([CA_Head(head_size, i) for i in range(num_heads)])

        # Projection layers for the head and body outputs
        self.proj_head = nn.Linear(head_size * num_heads, n_embd)
        self.proj_body = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_head, x_body):
        # Process inputs through all cross-attention heads
        all_outputs  = [h(x_head, x_body) for h in self.ca_heads]

        # Concatenate outputs from all cross-attention heads along the last dimension
        out_head = torch.cat([i[0] for i in all_outputs], dim=-1)
        out_body = torch.cat([i[1] for i in all_outputs], dim=-1)

        # Apply dropout and project the concatenated outputs back to the original embedding size
        out_head = self.dropout(self.proj_head(out_head))
        out_body = self.dropout(self.proj_body(out_body))
        return (out_head,out_body)
    
class Block(nn.Module):
    """Transformer block consisting of self-attention and cross-attention layers."""
    def __init__(self, n_embd, n_head, i):
        super().__init__()

        # Calculate head size based on embedding size and number of heads
        head_size = n_embd // n_head
        self.i = i

        # Initialize self-attention and cross-attention layers
        self.sa = MultiHeadSAAttention(n_head, head_size)
        self.ca = MultiHeadCAAttention(n_head, head_size)

        # Initialize feedforward layers and layer normalization
        self.ffwd_head = FeedForward(n_embd)
        self.ffwd_body = FeedForward(n_embd)
        self.ln1_head = nn.LayerNorm(n_embd)
        self.ln1_body = nn.LayerNorm(n_embd)
        self.ln2_head = nn.LayerNorm(n_embd)
        self.ln2_body = nn.LayerNorm(n_embd)

    def forward(self, x_full):
        # Separate head and body inputs
        x_head = x_full[0]
        x_body = x_full[1]
        
        # Self-Attention Block
        sa_x_head, sa_x_body = self.sa(x_head,x_body)
        x_head = x_head + self.ln1_head(sa_x_head)
        x_head = x_head + self.ln2_head(self.ffwd_head(x_head))
        x_body = x_body + self.ln1_body(sa_x_body)
        x_body = x_body + self.ln2_body(self.ffwd_body(x_body))

        # Cross-Attention Block
        ca_x_head, ca_x_body = self.ca(x_head,x_body)
        x_head = x_head + self.ln1_head(ca_x_head)
        x_head = x_head + self.ln2_head(self.ffwd_head(x_head))
        x_body = x_body + self.ln1_body(ca_x_body)
        x_body = x_body + self.ln2_body(self.ffwd_body(x_body))

        # Concatenate head and body outputs back into a single list
        x_full = [x_head, x_body]
        return x_full

class FeedForward(nn.Module):
    """Feedforward neural network layer for Transformer blocks."""

    def __init__(self, n_embd):
        super().__init__()
        
        # Define the feedforward network architecture
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        """Forward pass through the feedforward network"""
        return self.net(x)
    
class ClassificationHead(nn.Module):
    """Classification head for a neural network."""

    def __init__(self, input_size):
        super().__init__()

        # Define the classification network architecture
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
        """Forward pass through the classification network."""
        output = self.net(x)
        return output

class CrossEncoder(nn.Module):
    """CrossEncoder model for text classification."""

    def __init__(self, vocab_size, custom_embeddings=None):

        """
        Initializes the CrossEncoder model.

        Args:
        - vocab_size (int): The size of the vocabulary.
        - n_embd (int): The embedding size.
        - block_size (int): The size of the blocks in the Transformer architecture.
        - n_head (int): The number of attention heads in the Transformer blocks.
        - n_layer (int): The number of Transformer blocks.
        - custom_embeddings (torch.Tensor, optional): Pre-trained custom embeddings. Default is None.
        """

        super().__init__()

        # Flag to track if custom embeddings are used
        self.use_custom_embeddings = False
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        if custom_embeddings != None:
            self.use_custom_embeddings = True
            self.custom_embedding_table = nn.Embedding.from_pretrained(custom_embeddings, freeze=True)
            
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head, i=i) for i in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)         # Layer normalization for the final output
        self.classification_head = ClassificationHead(n_embd)         # Classification head for text classification
        self.apply(self._init_weights)         # Initialize weights

    def _init_weights(self, module):
        """
        Custom method to initialize weights of the neural network.

        Args:
        - module (nn.Module): The module to initialize weights for.
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx_head, idx_body):
        # Token embeddings
        tok_emb_head = self.token_embedding_table(idx_head)
        tok_emb_body = self.token_embedding_table(idx_body)

        # Positional embeddings
        pos_emb_head = self.position_embedding_table(torch.arange(idx_head.shape[1], device=device))
        pos_emb_body = self.position_embedding_table(torch.arange(idx_body.shape[1], device=device))

        if self.use_custom_embeddings:
            # Custom embeddings
            cus_emb_head = self.custom_embedding_table(idx_head)
            cus_emb_body = self.custom_embedding_table(idx_body)
            x_head = tok_emb_head + cus_emb_head + pos_emb_head
            x_body = tok_emb_body + cus_emb_body + pos_emb_body
        
        else:
            x_head = tok_emb_head + pos_emb_head
            x_body = tok_emb_body + pos_emb_body
        

        # Input for Transformer blocks
        x_full = [x_head, x_body]
        x_full = self.blocks(x_full)
        x_head = x_full[0]
        x_body = x_full[1]
        
        # Layer normalization
        x_head = self.ln_f(x_head)
        x_body = self.ln_f(x_body)

        # Pooling and classification
        pooled_output  = torch.cat([x_head.mean(dim=1), x_body.mean(dim=1)], dim=1)

        # Classification head
        logits = self.classification_head(pooled_output)
        return logits