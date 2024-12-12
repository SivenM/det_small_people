from typing import Optional
import torch
from torch import nn
from torch import Tensor
import numpy as np
from models import layers
from models import pos_emb_layers
from einops import rearrange

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


#######################################################
#######################################################
# Deformable modules


class DeformableEncoderBlock(nn.Module):

    def __init__(self, dpr, emb_dim=256, n_lvls=4, n_heads=8, n_points=4, dropout_prob=0.1) -> None:
        super().__init__()
        self.self_att = layers.MultiScaleDeformableAttention(emb_dim, n_lvls, n_heads, n_points)
        self.dropout_1 = nn.Dropout(dropout_prob)
        self.norm1 = nn.LayerNorm(emb_dim)

        self.ffn = layers.MLP(emb_dim)
        self.norm2 = nn.LayerNorm(emb_dim)
        self.dropout_2 = layers.StochasticDepth(dpr)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, features, pos, reference_points, spatial_shapes, lvl_start_idx):
        q = self.with_pos_embed(features, pos)
        features_2 = self.self_att(q, reference_points, features, spatial_shapes, lvl_start_idx)
        features = features + self.dropout_1(features_2)
        features = self.norm1(features)
        features_2 = self.ffn(features)
        features = features + self.dropout_2(features_2)
        features = self.norm2(features)
        return features
    

class DeformableEncoder(nn.Module):
    def __init__(self, emb_dim:int=256, num_blocks:int=1, stochastic_depth_rate:float=0.1):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_blocks = num_blocks
        self.stochastic_depth_rate = stochastic_depth_rate
        dpr = [x for x in np.linspace(0, stochastic_depth_rate, num_blocks)]
        modules = []
        for i in range(num_blocks):
            modules.append(DeformableEncoderBlock(dpr[i], emb_dim))
        self.t_blocks = nn.Sequential(*modules)

    @staticmethod
    def get_reference_points(spatial_shapes, device, valid_ratios=None):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):

            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            #ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            #ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        #reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self, x:Tensor, spatial_shapes:Tensor, lvl_start_idx:Tensor, pos=None):
        reference_points = self.get_reference_points(spatial_shapes, x.device)
        for block in self.t_blocks:
            x = block(x, pos, reference_points, spatial_shapes, lvl_start_idx)
        return x


class DeformableDecoderBlock(nn.Module):
    pass


class TimeSformerBlock(nn.Module):
    def __init__(self, 
                dpr,
                emb_dim:int=256,
                num_heads:int=8,
                hidden_dim:int=1024,
                dropout_brob:float=0.1,
                seed_device='cuda'
                ):
        super().__init__()
        self.spatial_norm = nn.LayerNorm(emb_dim)
        self.spatial_attention = layers.MultiHeadAttention(emb_dim, num_heads)
        self.drop1 = nn.Dropout(dropout_brob)

        self.temporal_norm = nn.LayerNorm(emb_dim)
        self.temporal_attention = layers.MultiHeadAttention(emb_dim, num_heads)
        self.temporal_fc = nn.Linear(emb_dim, emb_dim)
        self.drop2 = nn.Dropout(dropout_brob)

        self.ffn = layers.MLP(emb_dim, hidden_dim)
        self.norm_ffn = nn.LayerNorm(emb_dim)
        self.drop3 = layers.StochasticDepth(dpr, seed_device)

    def forward(self, x:Tensor, B, T, W):
        num_patches = x.size(1) // T
        H = num_patches // W

        xt = rearrange(x, 'b (h w t) m -> (b h w) t m', b=B, h=H, w=W, t=T)
        xt = self.temporal_norm(xt)
        xt = self.drop1(self.temporal_attention(xt,xt,xt))
        xt = rearrange(xt, '(b h w) t m -> b (h w t) m', b=B, h=H, w=W, t=T)
        xt = self.temporal_fc(xt)

        xs = rearrange(xt, 'b (h w t) m -> (b t) (h w) m', b=B, h=H, w=W, t=T)
        xs = self.spatial_norm(xs)
        xs = self.drop2(self.spatial_attention(xs, xs, xs))
        xs = rearrange(xs, '(b t) (h w) m -> b (h w t) m', b=B, h=H, w=W, t=T)

        x = xt + xs
        x = x + self.drop3(self.ffn(self.norm_ffn(x)))
        return x
    

class TimeSformer(nn.Module):
    def __init__(self, 
                num_blocks:int=5,
                emb_dim:int=256,
                patch_size:int=16,
                #num_output_channels:list=[64, 128],
                num_patches:int=1200,
                num_frames:int=10,
                num_heads:int=8,
                hidden_dim:int=1024,
                dropout_brob:float=0.1,
                stochastic_depth_rate:float=0.1,
                seed_device='cuda'
                ):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_blocks = num_blocks
        self.num_frames = num_frames
        self.stochastic_depth_rate = stochastic_depth_rate
        #self.patch_embed = layers.PatchEncoderSeqDeep(emb_dim, num_output_channels)
        self.patch_embed = layers.PatchEncoderSeq(emb_dim, patch_size)
        self.pos_embed = torch.nn.Parameter(torch.zeros(1, num_patches, emb_dim))
        self.time_embed = torch.nn.Parameter(torch.zeros(1, num_frames, emb_dim))
        self.drop_pos = nn.Dropout(dropout_brob)
        self.drop_time = nn.Dropout(dropout_brob)
        self.norm = nn.LayerNorm(emb_dim)

        dpr = [x for x in np.linspace(0, stochastic_depth_rate, num_blocks)]
        modules = []
        for i in range(num_blocks):
            modules.append(TimeSformerBlock(dpr[i], emb_dim, num_heads, hidden_dim, dropout_brob, seed_device))
        self.t_blocks = nn.Sequential(*modules)

    def forward(self, frame_seq:Tensor):
        batch_size = frame_seq.shape[0]
        x, width = self.patch_embed(frame_seq)
        x = x + self.pos_embed
        x = self.drop_pos(x)
        x = rearrange(x, '(b t) n m -> (b n) t m',b=batch_size,t=self.num_frames)
        x = x + self.time_embed
        x = self.drop_time(x)
        x = rearrange(x, '(b n) t m -> b (n t) m',b=batch_size,t=self.num_frames)
        for block in self.t_blocks:
            x = block(x,batch_size, self.num_frames, width)
        x = self.norm(x)
        return x