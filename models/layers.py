"""
Модуль, в котором хранятся слои
"""

from typing import Optional
import numpy as np
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, dilation=1, bias=True):
        super(SeparableConv2d, self).__init__()
        
        # Depthwise convolution
        self.depthwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=False
        )
        
        # Pointwise convolution (1x1 convolution)
        self.pointwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=bias
        )
        self.norm1 = nn.BatchNorm2d(in_channels)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.pointwise(x)
        x = self.norm2(x)
        return x


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
                SeparableConv2d(
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
                kernel_size=1,
                #stride=2
                ))
        modules.append(nn.ReLU())
        self.conv_model = nn.Sequential(*modules)

    def forward(self, images:Tensor) -> Tensor:
        batch = images.shape[0]
        features = self.conv_model(images)
        #print(features.shape)
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

# attention # 

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


class MultiScaleDeformableAttention(nn.Module):
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4):
        super().__init__()
        _d_per_head = d_model // n_heads

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

    def forward(self, query, reference_points, value, value_spatial_shapes):
        N, Len_q, _ = query.shape
        N, Len_in, _ = value.shape
        assert (value_spatial_shapes[:, 0] * value_spatial_shapes[:, 1]).sum() == Len_in

        value = self.value_proj(value)
        #value = value.masked_fill(value_padding_mask[..., None], float(0))
        value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)
        sampling_offsets = self.sampling_offsets(query).view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
        attention_weights = self.attention_weights(query).view(N, Len_q, self.n_heads, self.n_levels * self.n_points)
        attention_weights = F.softmax(attention_weights, -1).view(N, Len_q, self.n_heads, self.n_levels, self.n_points)
        offset_normalizer = torch.stack([value_spatial_shapes[..., 1], value_spatial_shapes[..., 0]], -1)
        sampling_locations = reference_points[:, :, None, :, None, :] + sampling_offsets / offset_normalizer[None, None, None, :, None, :]

        N_, _, M_, D_ = value.shape
        _, Lq_, M_, L_, P_, _ = sampling_locations.shape
        value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
        sampling_grids = 2 * sampling_locations - 1
        sampling_value_list = []
        for lid_, (H_, W_) in enumerate(value_spatial_shapes):
            # N_, H_*W_, M_, D_ -> N_, H_*W_, M_*D_ -> N_, M_*D_, H_*W_ -> N_*M_, D_, H_, W_
            value_l_ = value_list[lid_].flatten(2).transpose(1, 2).reshape(N_ * M_, D_, H_, W_)
            # N_, Lq_, M_, P_, 2 -> N_, M_, Lq_, P_, 2 -> N_*M_, Lq_, P_, 2
            sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1)
            # N_*M_, D_, Lq_, P_
            sampling_value_l_ = F.grid_sample(value_l_, sampling_grid_l_,
                                              mode='bilinear', padding_mode='zeros', align_corners=False)
            sampling_value_list.append(sampling_value_l_)
        # (N_, Lq_, M_, L_, P_) -> (N_, M_, Lq_, L_, P_) -> (N_, M_, 1, Lq_, L_*P_)
        attention_weights = attention_weights.transpose(1, 2).reshape(N_ * M_, 1, Lq_, L_ * P_)
        output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights).sum(-1).view(N_, M_ * D_, Lq_)
        output = output.transpose(1, 2).contiguous()

        output = self.output_proj(output)
        return output


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