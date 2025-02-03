import torch
from torch import nn


class LearnablePositionalEmbedding(nn.Module):
    def __init__(self, max_sequence_length, hidden_dim):
        super().__init__()
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, max_sequence_length, hidden_dim)
        )
        self.max_sequence_length = max_sequence_length
        nn.init.trunc_normal_(self.position_embeddings, std=0.02)

    def forward(self, x):
        bs = x.size(0)
        seq_length = x.size(1)
        if seq_length > self.max_sequence_length:
            raise ValueError(
                f"Sequence length ({seq_length}) exceeds maximum length ({self.max_sequence_length})"
            )

        position_embeddings = self.position_embeddings[:, :seq_length, :]
        position_embeddings = position_embeddings.expand(bs,-1,-1)
        return position_embeddings
    

class PositionEmbeddingLearned(nn.Module):

    def __init__(self, num_pos_feats=256):
        super().__init__()
        self.row_embed = nn.Embedding(100, num_pos_feats//2)
        self.col_embed = nn.Embedding(100, num_pos_feats//2)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, x):
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        return pos.flatten(2)
