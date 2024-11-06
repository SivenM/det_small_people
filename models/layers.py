"""
Модуль, в котором хранятся слои
"""

from typing import Optional
import numpy as np
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F


class PatchEncoderLinear(nn.Module):
    def __init__(self, num_patches, token_dim, emb_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection = torch.nn.Linear(token_dim, emb_dim)
        self.pos_emb = nn.Parameter(torch.randn(num_patches, emb_dim))

    def forward(self, patches):
        return self.projection(patches) + self.pos_emb
    

class CCTTokenaizer(nn.Module):

    def __init__(
            self, 
            emb_dim:int=256,  
            num_hidden_layers:int=2, 
            num_output_channels:list=[64, 128], 
            P:int=40, 
            C=1
            ):
        
        super().__init__()
        self.out_dim = num_output_channels[-1]
        self.patch_dim = P
        self.emb_dim = emb_dim
        modules = []
        inputs = C
        for i in range(num_hidden_layers):
            modules.append(
                nn.Conv2d(
                    inputs, 
                    num_output_channels[i],
                    kernel_size=3,
                    padding=1
                    ))
            modules.append(nn.ReLU())
            modules.append(nn.MaxPool2d(2))
            inputs = num_output_channels[i]
        modules.append(
            nn.Conv2d(
                inputs, 
                emb_dim,
                kernel_size=P,
                stride=P
                ))
        modules.append(nn.ReLU())
        self.conv_model = nn.Sequential(*modules)

    def forward(self, images:Tensor) -> Tensor:
        batch = images.shape[0]
        features = self.conv_model(images)
        f_size = features.shape[2:]
        num_patches = f_size[-1] * f_size[-2]
        patches = features.reshape(batch, num_patches, self.emb_dim)
        return patches
    

class PatchEncoderConv2D(nn.Module):
    def __init__(self, num_patches, emb_dim, P, C, pos=True):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_patches = num_patches
        self.conv_layer = nn.Conv2d(C, emb_dim, kernel_size=P, stride=P)
        self.pos = pos
        if pos:
            self.pos_emb = nn.Parameter(torch.randn(num_patches, emb_dim))
    
    def forward(self, img:Tensor) -> Tensor:
        batch = img.shape[0]
        patches = self.conv_layer(img)
        patches = patches.reshape(batch, self.emb_dim, self.num_patches).swapaxes(1,2)
        if self.pos:
            return patches + self.pos_emb
        else:
            return patches

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

    def forward(self, q, k, v):
        q = self.query(q)
        k = self.key(k)
        v = self.value(v)

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

    def forward(self, query, key, value):
        attention_scores = [attention(query, key, value) for attention in self.multi_head_attention]
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


class LocFFN(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class SeqPool(nn.Module):
    """
    Альтернатива AvgPool. Используй для классификации энкодера
    """
    def __init__(self, emb_dim:int=256) -> None:
        super().__init__()
        self.attention = nn.Linear(emb_dim, 1)
    
    def forward(self, x):
        bs = x.shape[0]
        att_weights = nn.functional.softmax(self.attention(x), dim=1)
        att_weights = att_weights.transpose(2,1)
        weighted_representation = torch.matmul(att_weights, x)
        return weighted_representation.reshape(bs, -1)
    

class StochasticDepth(nn.Module):
    """
    Dropout на уровне слоев блока
    https://arxiv.org/pdf/1603.09382
    """
    def __init__(self, drop_prob) -> None:
        super().__init__()
        self.drop_prob = drop_prob
        self.seed_generator = torch.Generator()
        self.seed_generator.manual_seed(1337)

    def forward(self, x):
        if self.training:
            keep_prob = 1 - self.drop_prob
            shape = (x.shape[0],) + (1,) * (len(x.shape) - 1)
            rand_tensor = keep_prob + torch.rand(shape, generator=self.seed_generator).to('cuda')
            return (x / keep_prob) * rand_tensor
        return x