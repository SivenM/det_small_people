import numpy as np
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from loguru import logger

class PatchEncoderLinear(nn.Module):
    def __init__(self, num_patches, token_dim, emb_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection = torch.nn.Linear(token_dim, emb_dim)
        self.pos_emb = nn.Parameter(torch.randn(num_patches, emb_dim))

    def forward(self, patches):
        return self.projection(patches) + self.pos_emb
    

class PatchEncoderConv2D(nn.Module):
    def __init__(self, num_patches, emb_dim, P, C):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_patches = num_patches
        self.conv_layer = nn.Conv2d(C, emb_dim, kernel_size=P, stride=P)
        self.pos_emb = nn.Parameter(torch.randn(num_patches, emb_dim))
    
    def forward(self, img:Tensor) -> Tensor:
        batch = img.shape[0]
        patches = self.conv_layer(img)
        patches = patches.reshape(batch, self.emb_dim, self.num_patches).swapaxes(1,2)
        return patches + self.pos_emb


class PatchEncoderPad(nn.Module):
    def __init__(self, patch_size:tuple, ):
        super().__init__()
        self.patch_size = patch_size

    def get_pad_h(self, h:int):
        assert h <= self.patch_size[0], f"Height ({h}) must be lower than patch_size {self.patch_size[0]}"
        if h == self.patch_size[0]:
            return (0,0)
        else:
            diff = self.patch_size[-1] - h
            if diff % 2 == 0:
                return (diff // 2, diff // 2)
            else:
                return (diff // 2, diff // 2 + 1)
        
    def get_pad_w(self, w:int):
        assert w <= self.patch_size[-1], f"Height ({w}) must be lower than patch_size {self.patch_size[-1]}"
        if w == self.patch_size[0]:
            return (0,0)
        else:
            diff = self.patch_size[-1] - w
            if diff % 2 == 0:
                return (diff // 2, diff // 2)
            else:
                return (diff // 2, diff // 2 + 1)
        
    def get_pad(self, img:Tensor, h:int, w:int):
        pad_h = self.get_pad_h(h)
        pad_w = self.get_pad_w(w)
        padded_img = F.pad(img, (pad_w[0], pad_w[1], pad_h[0], pad_h[1]), 'constant', 0)
        return padded_img
    
    def forward(self, img):
        _, _, h, w = img.shape
        img_padded = self.get_pad(img, h, w)
        return img_padded
    

class SelfAttention(nn.Module):
    def __init__(self, emb_dim=256, key_dim=64, dropout=0.0) -> None:
        super().__init__()
        self.emb_dim = emb_dim
        self.key_dim = key_dim

        self.query = nn.Linear(emb_dim, key_dim)
        self.key = nn.Linear(emb_dim, key_dim)
        self.value = nn.Linear(emb_dim, key_dim)

    def forward(self, x, memory=None):
        q = self.query(x)
        if memory != None:
            k = self.key(memory)
            v = self.value(memory)
        else:
            k = self.key(x)
            v = self.value(x)

        dot_prod = torch.matmul(q, torch.transpose(k, -2, -1))
        scaled_dot_prod = dot_prod / np.sqrt(self.key_dim)
        attention_weights = F.softmax(scaled_dot_prod, dim=1)
        weighted_values = torch.matmul(attention_weights, v)
        return weighted_values


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim=256, num_heads=8) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.emb_dim = emb_dim

        assert emb_dim % num_heads == 0, f'emb_dim: {emb_dim} | num_heads: {num_heads} | {emb_dim % num_heads}'
        self.key_dim = emb_dim // num_heads

        self.attention_list = [SelfAttention(emb_dim, self.key_dim) for _ in range(num_heads)]
        self.multi_head_attention = nn.ModuleList(self.attention_list)
        self.W = nn.Parameter(torch.randn(num_heads * self.key_dim, emb_dim))

    def forward(self, x, memory=None):
        if memory != None:
            attention_scores = [attention(x, memory) for attention in self.multi_head_attention]
        else:
            attention_scores = [attention(x) for attention in self.multi_head_attention]
        Z = torch.cat(attention_scores, -1)
        attention_score = torch.matmul(Z, self.W)
        return attention_score
    

class MLP(nn.Module):
    def __init__(self, emb_dim=256, hidden_dim=1024) -> None:
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, emb_dim)
        )

    def forward(self, x):
        x = self.mlp(x)
        return x
    

class TransformerBlock(nn.Module):
    def __init__(self, emb_dim=256, num_heads=8, hidden_dim=1024, dropout_prob=0.1) -> None:
        super().__init__()
        self.MSA = MultiHeadAttention(emb_dim, num_heads)
        self.MLP = MLP(emb_dim, hidden_dim)

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


class TransformerDecoder(nn.Module):

    def __init__(self, num_blocks:int, emb_dim=256, num_heads=8, hidden_dim=1024, dropout_prob=0.1) -> None:
        super().__init__()
        self.tfs = nn.Sequential(
            *[TransformerBlock(emb_dim, num_heads, hidden_dim, dropout_prob) for _ in range(num_blocks)]
        )
        self.ln1 = nn.LayerNorm(emb_dim)
        self.ln2 = nn.LayerNorm(emb_dim)
        self.ln3 = nn.LayerNorm(emb_dim)
        self.self_attn = MultiHeadAttention(emb_dim, num_heads)
        self.dropout1 = nn.Dropout(p=dropout_prob) 
        self.dropout2 = nn.Dropout(p=dropout_prob) 
        self.dropout3 = nn.Dropout(p=dropout_prob) 
        self.MSA = MultiHeadAttention(emb_dim, num_heads)
        self.mlp = MLP(emb_dim, hidden_dim)

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

        for tf in self.tfs:
            features = tf(features)

        return features
    

class DETR(nn.Module):
    
    def __init__(
            self, 
            num_encoder_blocks:int=6, 
            num_decoder_blocks:int=6, 
            num_queries:int=25,
            num_cls=1, 
            emb_dim=256,
            img_size=(480, 640),
            num_imgs = 10,
            patch_size=40
            ) -> None:
        super().__init__()

        num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)
        self.img_encoder= PatchEncoderConv2D(num_patches, emb_dim, patch_size, num_imgs)
        self.encoder = TransformerEncoder(num_encoder_blocks)
        self.decoder = TransformerDecoder(num_decoder_blocks)

        self.query_pos = nn.Parameter(torch.randn(num_queries, emb_dim))

        self.linear_class = nn.Linear(emb_dim, num_cls + 1)
        self.linear_bbox = nn.Linear(emb_dim, 4)

    def forward(self, x:Tensor) -> Tensor:
        features = self.img_encoder(x)
        features = self.encoder(features)
        features = self.decoder(self.query_pos, features)
        return {
            'logits': self.linear_class(features),
            'bbox': self.linear_bbox(features).sigmoid()
        }


class DetrLoc(nn.Module):
    """
    Предсказывает рамки людей на последнем фрейме (только локализация)
    """
    def __init__(
            self, 
            num_encoder_blocks:int=6, 
            num_decoder_blocks:int=6, 
            num_queries:int=25,
            num_cls=1, 
            emb_dim=256,
            img_size=(480, 640),
            num_imgs = 10,
            patch_size=40,
            loc_emb:list=[256, 128, 64]
            ) -> None:
        super().__init__()

        num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)
        self.img_encoder= PatchEncoderConv2D(num_patches, emb_dim, patch_size, num_imgs)
        self.encoder = TransformerEncoder(num_encoder_blocks)
        self.decoder = TransformerDecoder(num_decoder_blocks)

        self.query_pos = nn.Parameter(torch.randn(num_queries, emb_dim))

        #loc_layers = []
        #for le in loc_emb:
        #    loc_layers.append(nn.Linear(emb_dim, le))
        #    emb_dim = le
        #self.loc_layers = nn.Sequential(*loc_layers)

        self.out_head = nn.Linear(emb_dim, 4)

    def forward(self, x:Tensor) -> Tensor:
        features = self.img_encoder(x)
        features = self.encoder(features)
        features = self.decoder(self.query_pos, features)
        return self.out_head(features) 


class Sogmoid(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    def forward(self, x):
        return 1/(1+torch.exp(-x))


class VisTransformer(nn.Module):
    def __init__(self, num_blocks:int, encoder_type:str=None, num_cls:int=1) -> None:
        super().__init__()
        self.num_blocks = num_blocks
        self.num_cls = num_cls
        if encoder_type == 'pad':
            self.encoder = nn.Sequential(
                PatchEncoderPad((50,50)),
                PatchEncoderConv2D(100, 256, 5, 10)
            )
        else:
            self.encoder = PatchEncoderConv2D(100, 256, 5, 10)
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
