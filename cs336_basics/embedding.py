import torch
from torch import nn
import numpy as np

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        """
        Construct an embedding module.
        
        :param num_embeddings: int Size of the vocabulary
        :param embedding_dim: int Dimension of the embedding vectors, i.e., d_model
        :param device: torch.device | None = None Device to store the parameters on
        :param dtype: torch.dtype | None = None Data type of the parameters
        """
        super().__init__()
        self.num_embeddings: int = num_embeddings
        self.embedding_dim: int = embedding_dim
        self.device: torch.device = device
        self.dtype: torch.dtype = dtype
        self.embeddings = nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty(self.num_embeddings, self.embedding_dim),
                mean=0,
                std=1,
                a=-3,
                b=3
            )
        )
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Apply the linear transformation of the input"""
        # token_ids has shape (batch_size, sequence_length)
        # embeddings had shape (vocab_size, d_model)
        return self.embeddings[token_ids.long()]