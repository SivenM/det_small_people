from typing import Optional
import torch
from torch import nn
from torch import Tensor
import numpy as np
import layers
import pos_emb_layers

# old realization transformer #
###############################

class TransformerBlock(nn.Module):
    def __init__(self, emb_dim=256, num_heads=8, hidden_dim=1024, dropout_prob=0.1) -> None:
        super().__init__()
        self.MSA = layers.MultiHeadAttention(emb_dim, num_heads)
        self.MLP = layers.MLP(emb_dim, hidden_dim)

        self.layer_norm1 = nn.LayerNorm(emb_dim)
        self.layer_norm2 = nn.LayerNorm(emb_dim)

        self.dropout1 = nn.Dropout(p=dropout_prob) 
        self.dropout2 = nn.Dropout(p=dropout_prob) 
        self.dropout3 = nn.Dropout(p=dropout_prob) 

    def forward(self, x):
        out_drop_1 = self.dropout1(x)
        out_norm_1 = self.layer_norm1(out_drop_1)
        out_msa = self.MSA(out_norm_1)
        out_drop_2 = self.dropout2(out_msa)
        add = x + out_drop_2
        out_norm_2 = self.layer_norm2(add)
        out_mlp = self.MLP(out_norm_2)
        out_drop_3 = self.dropout3(out_mlp)
        return add + out_drop_3


class TransformerEncoder(nn.Module):

    def __init__(self, num_blocks:int, emb_dim=256, num_heads=8, hidden_dim=1024, dropout_prob=0.1) -> None:
        super().__init__()
        self.tfs = nn.Sequential(
            *[TransformerBlock(emb_dim, num_heads, hidden_dim, dropout_prob) for _ in range(num_blocks)]
        )

    def forward(self, inputs:Tensor):
        x = inputs
        for tf_block in self.tfs:
            x = tf_block(x)
        return x


class TfDecoderBlock(nn.Module):

    def __init__(self, emb_dim=256, num_heads=8, hidden_dim=1024, dropout_prob=0.1) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(emb_dim)
        self.ln2 = nn.LayerNorm(emb_dim)
        self.ln3 = nn.LayerNorm(emb_dim)
        self.self_attn = layers.MultiHeadAttention(emb_dim, num_heads)
        self.dropout1 = nn.Dropout(p=dropout_prob) 
        self.dropout2 = nn.Dropout(p=dropout_prob) 
        self.dropout3 = nn.Dropout(p=dropout_prob) 
        self.MSA = layers.MultiHeadAttention(emb_dim, num_heads)
        self.mlp = layers.MLP(emb_dim, hidden_dim)

    def forward(self, queries:Tensor, memory:Tensor):
        q = self.self_attn(queries) 
        queries = queries + self.dropout1(q)
        queries = self.ln1(queries)

        features = self.MSA(queries, memory)
        features = self.dropout2(features) + queries
        features1 = self.ln2(features)
        features = self.mlp(features1)
        features = self.dropout3(features) + features1
        features = self.ln3(features)
        return features


class TransformerDecoder(nn.Module):

    def __init__(self, num_blocks:int, emb_dim=256, num_heads=8, hidden_dim=1024, dropout_prob=0.1) -> None:
        super().__init__()
        self.tfs = nn.Sequential(
            *[TfDecoderBlock(emb_dim, num_heads, hidden_dim, dropout_prob) for _ in range(num_blocks)]
        )

    def forward(self, queries:Tensor, memory:Tensor):
        features = queries
        for tf in self.tfs:
            features = tf(features, memory)

        return features
    

class VisTransformer(nn.Module):
    def __init__(self, num_blocks:int, encoder_type:str=None, num_cls:int=1) -> None:
        super().__init__()
        self.num_blocks = num_blocks
        self.num_cls = num_cls
        if encoder_type == 'pad':
            self.encoder = nn.Sequential(
                layers.PatchEncoderPad((50,50)),
                layers.PatchEncoderConv2D(100, 256, 5, 10)
            )
        else:
            self.encoder = layers.PatchEncoderConv2D(100, 256, 5, 10)
        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock() for _ in range(num_blocks)]
        )
        self.glob_avg = nn.AdaptiveAvgPool2d((1,1))
        self.head = nn.Sequential(
            nn.Linear(1, 256),
            nn.GELU(),
            nn.Linear(256, num_cls),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        blocks_out = self.transformer_blocks(encoded)
        avg = self.glob_avg(blocks_out).squeeze(1)
        out = self.head(avg)
        return out.squeeze()


# CCT realizations#
###################

class CCTBlock(nn.Module):
    def __init__(self, dpr, emb_dim=256, num_heads=8, hidden_dim=1024, dropout_prob=0.1) -> None:
        super().__init__()
        self.MSA = layers.MultiHeadAttention(emb_dim, num_heads)
        self.MLP = layers.MLP(emb_dim, hidden_dim)

        self.layer_norm1 = nn.LayerNorm(emb_dim)
        self.layer_norm2 = nn.LayerNorm(emb_dim)

        self.dropout1 = nn.Dropout(p=dropout_prob) 
        self.dropout2 = nn.Dropout(p=dropout_prob) 
        self.dropout3 = layers.StochasticDepth(dpr)

    def forward(self, x):
        out_drop_1 = self.dropout1(x)
        out_norm_1 = self.layer_norm1(out_drop_1)
        out_msa = self.MSA(out_norm_1)
        out_drop_2 = self.dropout2(out_msa)
        add = x + out_drop_2
        out_norm_2 = self.layer_norm2(add)
        out_mlp = self.MLP(out_norm_2)
        out_drop_3 = self.dropout3(out_mlp)
        return add + out_drop_3



class CCTransformer(nn.Module):
    """
    Compact Transformer
    https://arxiv.org/abs/2104.05704
    """
    def __init__(
            self, 
            num_classes,
            emb_dim, 
            num_blocks,
            num_conv_layers,
            out_channel_outputs,
            patch_size,
            C,
            max_seq,
            stochastic_depth_rate=0.1
            ):
        super().__init__()
        self.img_encoder = layers.CCTTokenaizer(
            emb_dim,
            num_conv_layers,
            out_channel_outputs,
            P=patch_size,
            C=C,
        )
        self.pos_emb = pos_emb_layers.LearnablePositionalEmbedding(max_seq, emb_dim)
        dpr = [x for x in np.linspace(0, stochastic_depth_rate, num_blocks)]
        modules = []
        for i in range(num_blocks):
            modules.append(CCTBlock(dpr[i], emb_dim))
        self.t_blocks = nn.Sequential(*modules)
        self.norm = nn.LayerNorm(emb_dim)
        self.seq_pool = layers.SeqPool(emb_dim)
        self.logits = nn.Linear(emb_dim, num_classes)

    def forward(self, image:Tensor) -> Tensor:
        x = self.img_encoder(image)
        x = x + self.pos_emb(x)
        for block in self.t_blocks:
            x = block(x)
        x = self.norm(x)
        x = self.seq_pool(x)
        return nn.functional.sigmoid(self.logits(x))
    

class CCTransformerV2(nn.Module):
    def __init__(
            self, 
            num_classes,
            emb_dim, 
            num_blocks,
            num_conv_layers,
            out_channel_outputs,
            patch_size,
            C,
            max_seq,
            stochastic_depth_rate=0.1
            ):
        super().__init__()
        self.img_encoder = layers.CCTTokenaizer(
            emb_dim,
            num_conv_layers,
            out_channel_outputs,
            P=patch_size,
            C=C,
        )
        self.pos_emb = pos_emb_layers.LearnablePositionalEmbedding(max_seq, emb_dim)
        self.backbone = CCTEncoder(emb_dim, num_blocks, stochastic_depth_rate)
        self.norm = nn.LayerNorm(emb_dim)
        self.seq_pool = layers.SeqPool(emb_dim)
        self.logits = nn.Linear(emb_dim, num_classes)
    
    def forward(self, seq:Tensor):
        x = self.img_encoder(seq)
        pos = self.pos_emb(x)
        x = self.backbone(x, pos)
        x = self.norm(x)
        x = self.seq_pool(x)
        return nn.functional.sigmoid(self.logits(x))
    

class CCTEncoder(nn.Module):
    def __init__(self, emb_dim:int=256, num_blocks:int=1, stochastic_depth_rate:float=0.1):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_blocks = num_blocks
        self.stochastic_depth_rate = stochastic_depth_rate
        dpr = [x for x in np.linspace(0, stochastic_depth_rate, num_blocks)]
        modules = []
        for i in range(num_blocks):
            modules.append(CCTEncoderBlock(dpr[i], emb_dim))
        self.t_blocks = nn.Sequential(*modules)

    def forward(self, x:Tensor, pos=None):
        for block in self.t_blocks:
            x = block(x, pos)
        return x
        

class CCTDecoder(nn.Module):
    def __init__(self, emb_dim:int=256, num_blocks:int=1, stochastic_depth_rate:float=0.1):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_blocks = num_blocks
        self.stochastic_depth_rate = stochastic_depth_rate
        dpr = [x for x in np.linspace(0, stochastic_depth_rate, num_blocks)]
        modules = []
        for i in range(num_blocks):
            modules.append(CCTDecoderBlockV2(dpr[i], emb_dim))   #DEBUG
        self.t_blocks = nn.Sequential(*modules)

    def forward(self, query:Tensor, memory:Tensor,  pos=None, query_pos=None):
        for block in self.t_blocks:
            query = block(query, memory, pos, query_pos)
        return query


class CCTEncoderBlock(nn.Module):
    
    def __init__(self, dpr, emb_dim=256, num_heads=8, hidden_dim=1024, dropout_prob=0.1) -> None:
        super().__init__()
        self.MSA = layers.MultiHeadAttention(emb_dim, num_heads)
        self.MLP = layers.MLP(emb_dim, hidden_dim)

        self.layer_norm1 = nn.LayerNorm(emb_dim)
        self.layer_norm2 = nn.LayerNorm(emb_dim)

        self.dropout1 = nn.Dropout(p=dropout_prob) 
        self.dropout2 = layers.StochasticDepth(dpr)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, x: Tensor, pos: Optional[Tensor] = None):
        out_norm_1 = self.layer_norm1(x)
        q = k = self.with_pos_embed(out_norm_1, pos)
        out_msa = self.MSA(q, k, out_norm_1)
        out_drop_1 = self.dropout1(out_msa)
        x = x + out_drop_1
        out_norm_2 = self.layer_norm2(x)
        out_mlp = self.MLP(out_norm_2)
        out_drop_2 = self.dropout2(out_mlp)
        return x + out_drop_2


class CCTDecoderBlock(nn.Module):
    
    def __init__(self, dpr, emb_dim=256, num_heads=8, hidden_dim=2048, dropout_prob=0.1) -> None:
        super().__init__()
        self.self_att = layers.MultiHeadAttention(emb_dim, num_heads)
        self.MSA = layers.MultiHeadAttention(emb_dim, num_heads)
        self.MLP = layers.MLP(emb_dim, hidden_dim)

        self.layer_norm1 = nn.LayerNorm(emb_dim)
        self.layer_norm2 = nn.LayerNorm(emb_dim)
        self.layer_norm3 = nn.LayerNorm(emb_dim)

        self.dropout1 = nn.Dropout(p=dropout_prob) 
        self.dropout2 = nn.Dropout(p=dropout_prob) 
        self.dropout3 = layers.StochasticDepth(dpr)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, query: Tensor, memory:Tensor, pos: Optional[Tensor] = None, query_pos: Optional[Tensor] = None):
        query_norm = self.layer_norm1(query)
        q = k = self.with_pos_embed(query_norm, query_pos)
        query_norm = self.self_att(q, k, query_norm)
        out_drop_1 = self.dropout1(query_norm)
        query = query + out_drop_1
        query = self.layer_norm2(query)
        out_msa = self.MSA(
            self.with_pos_embed(query_norm, query_pos),
            self.with_pos_embed(memory, pos), 
            memory
            )
        out_drop_2 = self.dropout2(out_msa)
        query = query + out_drop_2
        out_norm_2 = self.layer_norm3(query)
        out_mlp = self.MLP(out_norm_2)
        out_drop_3 = self.dropout3(out_mlp)
        return query + out_drop_3


class CCTDecoderBlockV2(nn.Module):
    
    def __init__(self, dpr, emb_dim=256, num_heads=8, hidden_dim=1024, dropout_prob=0.1) -> None:
        super().__init__()
        self.self_att = nn.MultiheadAttention(emb_dim, num_heads, batch_first=True)
        self.MSA = nn.MultiheadAttention(emb_dim, num_heads, batch_first=True)
        self.MLP = layers.MLP(emb_dim, hidden_dim)

        self.layer_norm1 = nn.LayerNorm(emb_dim)
        self.layer_norm2 = nn.LayerNorm(emb_dim)
        self.layer_norm3 = nn.LayerNorm(emb_dim)

        self.dropout1 = nn.Dropout(p=dropout_prob) 
        self.dropout2 = nn.Dropout(p=dropout_prob) 
        self.dropout3 = layers.StochasticDepth(dpr)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, query: Tensor, memory:Tensor, pos: Optional[Tensor] = None, query_pos: Optional[Tensor] = None):
        query_norm = self.layer_norm1(query)
        q = k = self.with_pos_embed(query_norm, query_pos)
        query_norm = self.self_att(q, k, query_norm)[0]
        out_drop_1 = self.dropout1(query_norm)
        query = query + out_drop_1
        query = self.layer_norm2(query)
        out_msa = self.MSA(
            self.with_pos_embed(query_norm, query_pos),
            self.with_pos_embed(memory, pos), 
            memory
            )[0]
        out_drop_2 = self.dropout2(out_msa)
        query = query + out_drop_2
        out_norm_2 = self.layer_norm3(query)
        out_mlp = self.MLP(out_norm_2)
        out_drop_3 = self.dropout3(out_mlp)
        return query + out_drop_3
