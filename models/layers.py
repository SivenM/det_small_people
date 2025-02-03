"""
Модуль, в котором хранятся слои
"""

from typing import Optional
import math
import numpy as np
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_
from einops import rearrange


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
                nn.Conv2d(
                    inputs, 
                    num_output_channels[i],
                    kernel_size=3,
                    padding=1
                    ))
            modules.append(nn.ReLU())
            modules.append(nn.GroupNorm(8, num_output_channels[i]))
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


class DetrTokenizator(nn.Module):
    def __init__(self, C=1,
                 layers_dim=[32,64,128,256,512,1024]):
        super().__init__()
        self.layers_dim = layers_dim
        blocks= []
        for i in range(len(layers_dim)):
            if i == 0:
                conv = nn.Conv2d(C, layers_dim[i], kernel_size=3,padding=1)
            else:
                conv = nn.Conv2d(layers_dim[i-1], layers_dim[i], kernel_size=3,padding=1)
            blocks.append(
                nn.Sequential(
                    conv,
                    nn.ReLU(),
                    nn.GroupNorm(32, layers_dim[i]),
                    nn.MaxPool2d(2),
                )
            )
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x:Tensor):
        results = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i > 1:
                results.append(x)
        return results
    

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


class PatchEncoderSeq(nn.Module):
    """
    Patch encoder for frame sequence
    """
    def __init__(self,  
                 emb_dim:int=256,
                 patch_size:int=16,
                 in_channels:int=1,
                 ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, emb_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, seq_frames:Tensor):
        B, T, C, H, W = seq_frames.shape
        seq_frames = rearrange(seq_frames, 'b t c h w -> (b t) c h w')
        patches = self.conv(seq_frames)
        W = patches.size(-1)
        patches = patches.flatten(2).transpose(1,2) # ((b*t), num_p, emb_dim)
        return patches, W


class PatchEncoderSeqDeep(nn.Module):
    """
    Patch encoder for frame sequence
    """
    def __init__(self,  
                 emb_dim:int=256,
                 num_output_channels:list=[64, 128],
                 in_channels:int=1,
                 ) -> None:
        super().__init__()
        modules = []
        inputs = in_channels
        num_hidden_layers = len(num_output_channels)
        for i in range(num_hidden_layers):
            modules.append(
                nn.Conv2d(
                    inputs, 
                    num_output_channels[i],
                    kernel_size=3,
                    padding=1
                    ))
            modules.append(nn.ReLU())
            modules.append(nn.GroupNorm(8, num_output_channels[i]))
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

    def forward(self, seq_frames:Tensor):
        B, T, C, H, W = seq_frames.shape
        seq_frames = rearrange(seq_frames, 'b t c h w -> (b t) c h w')
        patches = self.conv_model(seq_frames)
        W = patches.size(-1)
        patches = patches.flatten(2).transpose(1,2) # ((b*t), num_p, emb_dim)
        return patches, W
    

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


class MSDeformAttn(nn.Module):
    def __init__(self, d_model=256, n_levels=4, n_heads=4, n_points=4):
        """
        Multi-Scale Deformable Attention Module
        :param d_model      hidden dimension
        :param n_levels     number of feature levels
        :param n_heads      number of attention heads
        :param n_points     number of sampling points per attention head per feature level
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads, but got {} and {}'.format(d_model, n_heads))
        self.head_dim = d_model // n_heads
        self.im2col_step = 64

        self.emb_dim = d_model
        self.num_levels = n_levels
        self.num_heads = n_heads
        self.num_points = n_points

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.num_heads, 1, 1, 2).repeat(1, self.num_levels, self.num_points, 1)
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def bilinear_interpolate(self, value, sampling_locations, spatial_shapes, level_start_index):
        N, Len_q, num_heads, num_points_total, _ = sampling_locations.shape
        _, Len_in, num_heads, head_dim = value.shape

        output = torch.zeros(N, Len_q, num_heads, num_points_total, head_dim, device=value.device)

        for level, (h, w) in enumerate(spatial_shapes):
            start_idx = level_start_index[level]
            end_idx = start_idx + h * w

            level_value = value[:, start_idx:end_idx, :, :]
            level_value = level_value.view(N, h, w, num_heads, head_dim)

            level_locations = sampling_locations[..., level * self.num_points:(level + 1) * self.num_points, :]
            level_locations = level_locations * torch.tensor([w, h], device=value.device).float() - 0.5

            x = level_locations[..., 0]  # (N, Len_q, num_heads, num_points)
            y = level_locations[..., 1]  # (N, Len_q, num_heads, num_points)

            x0 = torch.floor(x).long()
            x1 = x0 + 1
            y0 = torch.floor(y).long()
            y1 = y0 + 1

            x0 = torch.clamp(x0, 0, w - 1)
            x1 = torch.clamp(x1, 0, w - 1)
            y0 = torch.clamp(y0, 0, h - 1)
            y1 = torch.clamp(y1, 0, h - 1)

            wa = (x1 - x) * (y1 - y)  # (N, Len_q, num_heads, num_points)
            wb = (x1 - x) * (y - y0)  # (N, Len_q, num_heads, num_points)
            wc = (x - x0) * (y1 - y)  # (N, Len_q, num_heads, num_points)
            wd = (x - x0) * (y - y0)  # (N, Len_q, num_heads, num_points)

            # Добавляем измерение head_dim для согласования размерностей
            wa = wa[..., None]  # (N, Len_q, num_heads, num_points, 1)
            wb = wb[..., None]  # (N, Len_q, num_heads, num_points, 1)
            wc = wc[..., None]  # (N, Len_q, num_heads, num_points, 1)
            wd = wd[..., None]  # (N, Len_q, num_heads, num_points, 1)

            # Извлекаем значения для каждой точки
            value_a = level_value[:, y0, x0, :, :]  # (N, Len_q, num_heads, head_dim)
            value_b = level_value[:, y1, x0, :, :]  # (N, Len_q, num_heads, head_dim)
            value_c = level_value[:, y0, x1, :, :]  # (N, Len_q, num_heads, head_dim)
            value_d = level_value[:, y1, x1, :, :]  # (N, Len_q, num_heads, head_dim)

            # Приводим value_a, value_b, value_c, value_d к размерности (N, Len_q, num_heads, num_points, head_dim)
            value_a = value_a.unsqueeze(3).expand(-1, -1, -1, self.num_points, -1)
            value_b = value_b.unsqueeze(3).expand(-1, -1, -1, self.num_points, -1)
            value_c = value_c.unsqueeze(3).expand(-1, -1, -1, self.num_points, -1)
            value_d = value_d.unsqueeze(3).expand(-1, -1, -1, self.num_points, -1)

            # Взвешенное суммирование
            output[:, :, :, level * self.num_points:(level + 1) * self.num_points, :] += (
                wa * value_a +
                wb * value_b +
                wc * value_c +
                wd * value_d
            )

        return output


    def forward(self, query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index):
        """
        :param query                       (N, Length_{query}, C)
        :param reference_points            (N, Length_{query}, n_levels, 2), range in [0, 1], top-left (0,0), bottom-right (1, 1), including padding area
                                        or (N, Length_{query}, n_levels, 4), add additional (w, h) to form reference boxes
        :param input_flatten               (N, \sum_{l=0}^{L-1} H_l \cdot W_l, C)
        :param input_spatial_shapes        (n_levels, 2), [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
        :param input_level_start_index     (n_levels, ), [0, H_0*W_0, H_0*W_0+H_1*W_1, H_0*W_0+H_1*W_1+H_2*W_2, ..., H_0*W_0+H_1*W_1+...+H_{L-1}*W_{L-1}]
        :param input_padding_mask          (N, \sum_{l=0}^{L-1} H_l \cdot W_l), True for padding elements, False for non-padding elements

        :return output                     (N, Length_{query}, C)
        """
        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape
        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in

        value = self.value_proj(input_flatten)
        value = value.view(N, Len_in, self.num_heads, self.head_dim)

        sampling_offsets = self.sampling_offsets(query).view(N, Len_q, self.num_heads, self.num_levels, self.num_points, 2)
        attention_weights = self.attention_weights(query).view(N, Len_q, self.num_heads, self.num_levels * self.num_points)
        attention_weights = F.softmax(attention_weights, -1).view(N, Len_q, self.num_heads, self.num_levels, self.num_points)

        # Сборка смещений и применение их к reference_points
        N, Len_q, num_heads, num_levels, num_points, _ = sampling_offsets.shape
        reference_points = reference_points[:, :, None, :, None, :]
        sampling_locations = reference_points + sampling_offsets

        # Билинейная интерполяция для получения значений
        sampling_locations = sampling_locations.view(N, Len_q, num_heads, num_levels * num_points, 2)
        sampling_locations = sampling_locations.clamp(min=0, max=1)  # Ограничение в пределах [0, 1]
        sampled_values = self.bilinear_interpolate(value, sampling_locations, input_spatial_shapes, input_level_start_index)

        # Взвешенное суммирование
        output = torch.einsum('nlhpd,nlhpd->nlhd', attention_weights, sampled_values)
        output = output.view(N, Len_q, self.emb_dim)
        output = self.output_proj(output)

        return output


class DeformableAttention(nn.Module):
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4):
        """
        Инициализация Deformable Attention.

        :param d_model: Размерность входных и выходных тензоров.
        :param n_levels: Количество уровней признаков (например, разные разрешения特征 pyramid).
        :param n_heads: Количество голов внимания.
        :param n_points: Количество точек отсчета для каждой головы.
        """
        super(DeformableAttention, self).__init__()
        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

        # Убедимся, что размерность модели делится на количество голов
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.head_dim = d_model // n_heads

        # Линейные слои для преобразования query, key и value
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)

        # Слои для вычисления смещений точек отсчета
        self.sampling_loc = nn.Parameter(torch.zeros(n_heads, n_levels, n_points, 2))
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)

        # Выходной линейный слой
        self.output_proj = nn.Linear(d_model, d_model)

    def forward(self, query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index):
        """
        Прямой проход через Deformable Attention.

        :param query: Tensor [B, N_q, C], где B - размер батча, N_q - количество запросов, C - размерность канала.
        :param reference_points: Tensor [B, N_q, n_levels, 2], нормализованные координаты точек отсчета.
        :param input_flatten: Tensor [B, \sum_{l=0}^{L-1} H_l \times W_l, C], сплющенные признаки из разных уровней.
        :param input_spatial_shapes: Tensor [n_levels, 2], формы пространственных размерностей для каждого уровня.
        :param input_level_start_index: Tensor [n_levels], индексы начала каждого уровня в input_flatten.
        :return: Tensor [B, N_q, C], результат внимания.
        """
        B, N_q, C = query.shape
        _, N, _ = input_flatten.shape

        # Проектируем query, key и value
        query = self.query_proj(query)  # [B, N_q, C]
        key = self.key_proj(input_flatten)  # [B, N, C]
        value = self.value_proj(input_flatten)  # [B, N, C]

        # Разделяем по головам
        query = query.view(B, N_q, self.n_heads, self.head_dim).transpose(1, 2)  # [B, n_heads, N_q, head_dim]
        key = key.view(B, N, self.n_heads, self.head_dim).transpose(1, 2)  # [B, n_heads, N, head_dim]
        value = value.view(B, N, self.n_heads, self.head_dim).transpose(1, 2)  # [B, n_heads, N, head_dim]

        # Вычисляем веса внимания
        attention_weights = self.attention_weights(query).view(
            B, N_q, self.n_heads, self.n_levels, self.n_points
        )  # [B, N_q, n_heads, n_levels, n_points]
        attention_weights = F.softmax(attention_weights, dim=-1).view(
            B, N_q, self.n_heads, self.n_levels * self.n_points
        )  # [B, N_q, n_heads, n_levels * n_points]

        # Вычисляем позиции отсчета
        sampling_offsets = self.sampling_loc.unsqueeze(0).repeat(B, 1, 1, 1, 1)  # [B, n_heads, n_levels, n_points, 2]
        reference_points = reference_points[:, :, None, :, None, :]  # [B, N_q, 1, n_levels, 1, 2]
        sampling_locations = reference_points + sampling_offsets  # [B, N_q, n_heads, n_levels, n_points, 2]

        # Преобразуем sampling_locations в индексы для grid_sample
        sampling_grids = 2 * sampling_locations - 1  # Нормализация в диапазон [-1, 1]

        # Вычисляем значения ключей и значений в точках отсчета
        value_samples = []
        for level in range(self.n_levels):
            start_idx = input_level_start_index[level]
            end_idx = input_level_start_index[level + 1] if level < self.n_levels - 1 else N
            curr_value = value[:, :, start_idx:end_idx].permute(0, 3, 1, 2)  # [B, head_dim, n_heads, H_l*W_l]
            H_l, W_l = input_spatial_shapes[level]
            curr_value = curr_value.view(B, self.head_dim, self.n_heads, H_l, W_l)  # [B, head_dim, n_heads, H_l, W_l]
            curr_value = curr_value.permute(0, 2, 1, 3, 4).flatten(0, 1)  # [B*n_heads, head_dim, H_l, W_l]

            # Применяем grid_sample для интерполяции значений
            sampling_grid_lvl = sampling_grids[:, :, :, level].flatten(0, 1)  # [B*N_q*n_heads, n_points, 2]
            sampled_value = F.grid_sample(
                curr_value,
                sampling_grid_lvl[:, None],
                mode='bilinear',
                padding_mode='zeros',
                align_corners=False
            )[:, :, 0, :].view(B, self.n_heads, N_q, self.n_points, self.head_dim)  # [B, n_heads, N_q, n_points, head_dim]

            value_samples.append(sampled_value)

        value_samples = torch.stack(value_samples, dim=3)  # [B, n_heads, N_q, n_levels, n_points, head_dim]
        value_samples = value_samples.flatten(3, 4)  # [B, n_heads, N_q, n_levels * n_points, head_dim]

        # Вычисляем внимание
        output = (value_samples * attention_weights.unsqueeze(-1)).sum(dim=3)  # [B, n_heads, N_q, head_dim]

        # Объединяем головы
        output = output.transpose(1, 2).contiguous().view(B, N_q, C)  # [B, N_q, C]

        # Проецируем выход
        output = self.output_proj(output)  # [B, N_q, C]

        return output


class DfAttn(nn.Module):

    def __init__(self, d_model=256, n_levels=4, n_heads=4, n_points=4):
        """
        Multi-Scale Deformable Attention Module
        :param d_model      hidden dimension
        :param n_levels     number of feature levels
        :param n_heads      number of attention heads
        :param n_points     number of sampling points per attention head per feature level
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads, but got {} and {}'.format(d_model, n_heads))
        self.head_dim = d_model // n_heads
        self.im2col_step = 64

        self.emb_dim = d_model
        self.num_levels = n_levels
        self.num_heads = n_heads
        self.num_points = n_points

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

    def forward(self, query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index):
        """
        :param query                       (N, Length_{query}, C)
        :param reference_points            (N, Length_{query}, n_levels, 2), range in [0, 1], top-left (0,0), bottom-right (1, 1), including padding area
                                        or (N, Length_{query}, n_levels, 4), add additional (w, h) to form reference boxes
        :param input_flatten               (N, \sum_{l=0}^{L-1} H_l \cdot W_l, C)
        :param input_spatial_shapes        (n_levels, 2), [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
        :param input_level_start_index     (n_levels, ), [0, H_0*W_0, H_0*W_0+H_1*W_1, H_0*W_0+H_1*W_1+H_2*W_2, ..., H_0*W_0+H_1*W_1+...+H_{L-1}*W_{L-1}]
        :param input_padding_mask          (N, \sum_{l=0}^{L-1} H_l \cdot W_l), True for padding elements, False for non-padding elements

        :return output                     (N, Length_{query}, C)
        """
        N, Len_q, C = query.shape
        N, Len_in, _ = input_flatten.shape
        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in

        value = self.value_proj(input_flatten)
        value = value.view(N, Len_in, self.num_heads, self.head_dim)

        sampling_offsets = self.sampling_offsets(query).view(N, Len_q, self.num_heads, self.num_levels, self.num_points, 2)
        attention_weights = self.attention_weights(query).view(N, Len_q, self.num_heads, self.num_levels * self.num_points)
        attention_weights = F.softmax(attention_weights, -1).view(N, Len_q, self.num_heads, self.num_levels, self.num_points)

        # Сборка смещений и применение их к reference_points
        N, Len_q, num_heads, num_levels, num_points, _ = sampling_offsets.shape
        reference_points = reference_points[:, :, None, :, None, :] # (N, Len_q, 1, num_levels, 1, 2)
        print(f'ref_points: {reference_points.shape}')
        print(f'sampling offsets: {sampling_offsets.shape}')
        sampling_locations = reference_points + sampling_offsets # (N, Len_q, num_heads, num_levels, num_points, 2)

        # Преобразуем sampling_locations в индексы для grid_sample
        sampling_grids = 2 * sampling_locations - 1  # Нормализация в диапазон [-1, 1]
        N_, _, M_, D_ = value.shape
        _, Lq_, M_, L_, P_, _ = sampling_locations.shape
        value_list = value.split([H_ * W_ for H_, W_ in input_spatial_shapes], dim=1)
        #for val in value_list:
        #    print(f'value list val: {val.shape}')
        #print('\n\n')
        sampling_grids = 2 * sampling_locations - 1
        sampling_value_list = []
        for lid_, (H_, W_) in enumerate(input_spatial_shapes):
            # N_, H_*W_, M_, D_ -> N_, H_*W_, M_*D_ -> N_, M_*D_, H_*W_ -> N_*M_, D_, H_, W_
            value_l_ = value_list[lid_].flatten(2).transpose(1, 2).reshape(N_ * M_, D_, H_, W_)
            #print(f'value_l: {value_l_.shape}')
            # N_, Lq_, M_, P_, 2 -> N_, M_, Lq_, P_, 2 -> N_*M_, Lq_, P_, 2
            sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1)
            #print(f'{sampling_grid_l_.shape=}')
            # N_*M_, D_, Lq_, P_
            sampling_value_l_ = F.grid_sample(value_l_, sampling_grid_l_,
                                              mode='bilinear', padding_mode='zeros', align_corners=False)
            sampling_value_list.append(sampling_value_l_)
            #print('\n')
        # (N_, Lq_, M_, L_, P_) -> (N_, M_, Lq_, L_, P_) -> (N_, M_, 1, Lq_, L_*P_)
        attention_weights = attention_weights.transpose(1, 2).reshape(N_ * M_, 1, Lq_, L_ * P_)
        output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights).sum(-1).view(N_, M_ * D_, Lq_)
        output = output.transpose(1, 2).contiguous()

        output = self.output_proj(output)
        return output


class MLP(nn.Module):
    def __init__(self, emb_dim=256, hidden_dim=1024, activation='relu') -> None:
        super().__init__()
        if activation == 'relu':
            self.act = nn.ReLU()
        elif activation == 'gelu':
            self.act = nn.GELU()
        else:
            raise 'wrong activatoin!'
        
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            self.act,
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
    def __init__(self, drop_prob, seed_device='cuda') -> None:
        super().__init__()
        self.drop_prob = drop_prob
        self.seed_device = seed_device
        self.seed_generator = torch.Generator()
        self.seed_generator.manual_seed(1337)

    def forward(self, x):
        if self.training:
            keep_prob = 1 - self.drop_prob
            shape = (x.shape[0],) + (1,) * (len(x.shape) - 1)
            rand_tensor = keep_prob + torch.rand(shape, generator=self.seed_generator).to(self.seed_device)
            return (x / keep_prob) * rand_tensor
        return x
    


def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError("invalid input for _is_power_of_2: {} (type: {})".format(n, type(n)))
    return (n & (n-1) == 0) and n != 0