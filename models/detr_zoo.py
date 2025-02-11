import torch
from torch import nn
from torch import Tensor
from torchvision.models import resnet50
from models import layers
from models import pos_emb_layers
from models import transformer
import copy

# base mopdels #

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
        self.img_encoder= layers.PatchEncoderConv2D(num_patches, emb_dim, patch_size, num_imgs)
        self.encoder = transformer.TransformerEncoder(num_encoder_blocks)
        self.decoder = transformer.TransformerDecoder(num_decoder_blocks)

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
        self.img_encoder= layers.PatchEncoderConv2D(num_patches, emb_dim, patch_size, num_imgs)
        self.encoder = transformer.TransformerEncoder(num_encoder_blocks)
        self.decoder = transformer.TransformerDecoder(num_decoder_blocks)

        self.query_pos = nn.Parameter(torch.randn(num_queries, emb_dim))

        #loc_layers = []
        #for le in loc_emb:
        #    loc_layers.append(nn.Linear(emb_dim, le))
        #    emb_dim = le
        #self.loc_layers = nn.Sequential(*loc_layers)
        #self.loc1 = nn.Linear(emb_dim, 512)
        #self.gelu1 = nn.GELU()
        #self.loc2 = nn.Linear(512, 128)
        #self.gelu2 = nn.GELU()
        self.out_head = nn.Linear(emb_dim, 4)

    def forward(self, x:Tensor) -> Tensor:
        features = self.img_encoder(x)
        features = self.encoder(features)
        features = self.decoder(self.query_pos, features)
        
        #features = self.loc1(features)
        #features = self.gelu1(features)
        #features = self.loc2(features)
        #features = self.gelu2(features)
        return self.out_head(features)#.sigmoid()


class SeqDetrLoc(nn.Module):
    def __init__(
            self, 
            emb_dim:int=256,
            num_bboxes:int=10,
            pos_per_block:bool=True,
            num_encoder_blocks:int=1,
            num_decoder_blocks:int=1,
            num_conv_layers:int=2,
            out_channel_outputs:list=[64, 128],
            patch_size:int=5,
            C:int=10,
            max_seq:int=768,
            stochastic_depth_rate=0.1
            
            ):
        super().__init__()
        self.pos_per_block = pos_per_block
        self.tokenizer = layers.CCTTokenaizer(
            emb_dim,
            num_conv_layers,
            out_channel_outputs,
            P=patch_size,
            C=C,
        )
        self.pos_emb = pos_emb_layers.LearnablePositionalEmbedding(max_seq, emb_dim)
        self.encoder = transformer.CCTEncoder(emb_dim, num_encoder_blocks, stochastic_depth_rate)
        self.decoder = transformer.CCTDecoder(emb_dim, num_decoder_blocks, stochastic_depth_rate)
        self.query_emb = nn.Parameter(torch.rand(1, num_bboxes, emb_dim), requires_grad=True)
        self.bbox_embed = layers.LocFFN(emb_dim, emb_dim, 4, 3)
        self.sigma = nn.Sigmoid()

    def forward(self, sequence:Tensor) -> Tensor:
        tokens = self.tokenizer(sequence)
        pos = self.pos_emb(tokens)
        if self.pos_per_block:
            features = self.encoder(tokens, pos)
            q = self.query_emb.repeat(len(sequence), 1, 1)
            out = self.decoder(torch.zeros_like(q), features, pos, q)
        else:
            features = self.encoder(tokens + pos)
            q = self.query_emb.repeat(len(sequence), 1, 1)
            out = self.decoder(q, features)
        bboxes = self.bbox_embed(out)
        return self.sigma(bboxes) 


class CCTransformerLoc(nn.Module):
    def __init__(
            self, 
            num_bboxes,
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
        self.backbone = transformer.CCTEncoder(emb_dim, num_blocks, stochastic_depth_rate)
        self.norm = nn.LayerNorm(emb_dim)
        self.seq_pool = layers.SeqPool(emb_dim)
        
        #loc head
        self.cx = nn.Linear(emb_dim, num_bboxes)
        self.cy = nn.Linear(emb_dim, num_bboxes)
        self.w = nn.Linear(emb_dim, num_bboxes)
        self.h = nn.Linear(emb_dim, num_bboxes)
    
    def forward(self, seq:Tensor):
        x = self.img_encoder(seq)
        pos = self.pos_emb(x)
        x = self.backbone(x, pos)
        x = self.norm(x)
        x = self.seq_pool(x)
        
        cx = self.cx(x)
        cy = self.cy(x)
        w = self.w(x)
        h = self.h(x)
        bboxes = torch.stack([cx,cy,w,h])
        bboxes = bboxes.permute(1,2,0)
        return bboxes.sigmoid()


class CCTransformerDet(nn.Module):
    def __init__(
            self, 
            num_bboxes,
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
        self.backbone = transformer.CCTEncoder(emb_dim, num_blocks, stochastic_depth_rate)
        self.norm = nn.LayerNorm(emb_dim)
        self.seq_pool = layers.SeqPool(emb_dim)
        
        #loc head
        self.cx = nn.Linear(emb_dim, num_bboxes)
        self.cy = nn.Linear(emb_dim, num_bboxes)
        self.w = nn.Linear(emb_dim, num_bboxes)
        self.h = nn.Linear(emb_dim, num_bboxes)

        #cls head
        self.cls_head = nn.Linear(emb_dim, num_bboxes)

    def forward(self, seq:Tensor): # (bs, 10, 480, 640)
        x = self.img_encoder(seq)
        pos = self.pos_emb(x)
        x = self.backbone(x, pos)
        x = self.norm(x)
        x = self.seq_pool(x)
        
        cx = self.cx(x)
        cy = self.cy(x)
        w = self.w(x)
        h = self.h(x)
        bboxes = torch.stack([cx,cy,w,h])
        bboxes = bboxes.permute(1,2,0)
        bboxes = bboxes.sigmoid()

        logits = self.cls_head(x)
        return {'bbox': bboxes, 'logits': logits} 
    

class TimeSformerDet(nn.Module):
    def __init__(self,
                num_bboxes:int=7,
                emb_dim:int=256,
                num_blocks:int=5,
                #patch_size:int=16,
                num_output_channels:list=[64, 128],
                num_patches:int=1200,
                num_frames:int=10,
                num_heads:int=8,
                hidden_dim:int=1024,
                dropout_brob:float=0.1,
                stochastic_depth_rate:float=0.1,
                seed_device='cuda' 
                ):
        super().__init__()
        self.transformer = transformer.TimeSformer(
            num_blocks,
            emb_dim,
            num_output_channels,
            num_patches,
            num_frames,
            num_heads,
            hidden_dim,
            dropout_brob,
            stochastic_depth_rate,
            seed_device,
        )

        self.seq_pool = layers.SeqPool(emb_dim)

        #loc head
        self.cx = nn.Linear(emb_dim, num_bboxes)
        self.cy = nn.Linear(emb_dim, num_bboxes)
        self.w = nn.Linear(emb_dim, num_bboxes)
        self.h = nn.Linear(emb_dim, num_bboxes)
        #cls head
        self.cls_head = nn.Linear(emb_dim, num_bboxes)

    def forward(self, seq):
        feat = self.transformer(seq)
        feat = self.seq_pool(feat)
        
        cx = self.cx(feat)
        cy = self.cy(feat)
        w = self.w(feat)
        h = self.h(feat)
        bboxes = torch.stack([cx,cy,w,h])
        bboxes = bboxes.permute(1,2,0)
        bboxes = bboxes.sigmoid()

        logits = self.cls_head(feat)
        return {'bbox': bboxes, 'logits': logits}
    

######################################################################
# vanila detr #
####################################################################


class VanillaDETR(nn.Module):
    def __init__(self, hidden_dim, nheads,
        num_encoder_layers, num_decoder_layers):
        super().__init__()
        # We take only convolutional layers from ResNet-50 model
        self.backbone = nn.Sequential(*list(resnet50(pretrained=True).children())[:-2])
        self.conv = nn.Conv2d(2048, hidden_dim, 1)
        self.transformer = nn.Transformer(hidden_dim, nheads,
        num_encoder_layers, num_decoder_layers)
    #    self.linear_class = nn.Linear(hidden_dim, num_classes + 1)
        self.linear_bbox = nn.Linear(hidden_dim, 4)
        self.query_pos = nn.Parameter(torch.rand(100, hidden_dim))
        self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        print(f'col emb: {self.col_embed.shape}\nrow emb: {self.row_embed.shape}')
    
    def forward(self, inputs):
        x = self.backbone(inputs)
        print(f'resnet out: {x.shape}')
        h = self.conv(x)
        print(f'conv out: {h.shape}')
        H, W = h.shape[-2:]
        pos = torch.cat([
            self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
            self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
            ], dim=-1).flatten(0, 1).unsqueeze(1).repeat(1, h.shape[0], 1)
        #print(f'col emb squeezed: {self.col_embed[:W].unsqueeze(0).shape}')
        #print(f'row emb squeezed: {self.row_embed[:H].unsqueeze(1).shape}')
        #print(f'col emb repeted: {self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1).shape}')
        #print(f'row emb repeted: {self.row_embed[:H].unsqueeze(0).repeat(1, W, 1).shape}')
        #print(f'pos: {torch.cat([
        #self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
        #self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
        #], dim=-1).shape}')
        #print(f'pos shape: {torch.cat([
        #self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
        #self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
        #], dim=-1).flatten(0, 1).shape}')
        #print(f'pos shape: {pos.shape}')
        #print(f'conv flattened: {h.flatten(2).shape}')
        #print(f'h: {h.flatten(2).permute(2, 0, 1).shape}')
        h_flattened = h.flatten(2).permute(2, 0, 1)
        print(f'pos shape {pos.shape}\nh flattened: {h_flattened.shape}')
        s = pos + h_flattened
        print(f's: ')
        h = self.transformer(pos + h_flattened, self.query_pos.unsqueeze(1).repeat(1, h.shape[0], 1))
        return self.linear_bbox(h).sigmoid()
    

class MyVanilaDetr(nn.Module):
    def __init__(self, hidden_dim, nheads,
        num_encoder_layers, num_decoder_layers):
        super().__init__()
        self.backbone = nn.Sequential(*list(resnet50(pretrained=True).children())[:-2])
        for layer in self.backbone.parameters():
            layer.requires_grad = False
        self.conv = nn.Conv2d(2048, hidden_dim, 1)
        self.transformer = nn.Transformer(hidden_dim, nheads, num_encoder_layers, num_decoder_layers)
        self.bbox_embed = layers.LocFFN(hidden_dim, hidden_dim, 4, 3)
        self.query_pos = nn.Parameter(torch.rand(10, hidden_dim))
        self.pos_emb = pos_emb_layers.PositionEmbeddingLearned(hidden_dim)

    def forward(self, inputs):
        x = self.backbone(inputs)
        #print(f'resnet out: {x.shape}')
        h = self.conv(x)
        #print(f'conv out: {h.shape}')
        pos = self.pos_emb(h)
        #print(f'pos: {pos.shape}')
        h_flattened = h.flatten(2).permute(2, 0, 1)
        pos = pos.permute(2,0,1)
        #print(f'pos shape {pos.shape}\nh flattened: {h_flattened.shape}')
        h = self.transformer(pos + h_flattened, self.query_pos.unsqueeze(1).repeat(1, h.shape[0], 1))
        #print(f'out transformer: {h.shape}')
        #print(f'out transposed: {h.transpose(0, 1).shape}')
        return self.bbox_embed(h.transpose(0, 1)).sigmoid()
    

class MyVanilaDetrV2(nn.Module):
    def __init__(self, hidden_dim, nheads,
        num_encoder_layers, num_decoder_layers):
        super().__init__()
        self.backbone = nn.Sequential(*list(resnet50(pretrained=True).children())[:-2])
        for layer in self.backbone.parameters():
            layer.requires_grad = False
        self.conv = nn.Conv2d(2048, hidden_dim, 1)
        self.transformer = nn.Transformer(hidden_dim, nheads, num_encoder_layers, num_decoder_layers)
        self.bbox_embed = layers.LocFFN(hidden_dim, hidden_dim, 4, 3)
        self.query_pos = nn.Parameter(torch.rand(10, hidden_dim))
        self.pos_emb = pos_emb_layers.PositionEmbeddingLearned(hidden_dim)

    def forward(self, inputs):
        x = self.backbone(inputs)
        #print(f'resnet out: {x.shape}')
        h = self.conv(x)
        #print(f'conv out: {h.shape}')
        pos = self.pos_emb(h)
        #print(f'pos: {pos.shape}')
        h_flattened = h.flatten(2).permute(2, 0, 1)
        pos = pos.permute(2,0,1)
        #print(f'pos shape {pos.shape}\nh flattened: {h_flattened.shape}')
        h = self.transformer(pos + h_flattened, self.query_pos.unsqueeze(1).repeat(1, h.shape[0], 1))
        #print(f'out transformer: {h.shape}')
        #print(f'out transposed: {h.transpose(0, 1).shape}')
        return self.bbox_embed(h.transpose(0, 1)).sigmoid()



#####################
# DefDETR
#####################


class DeformableDETR(nn.Module):
    def __init__(self, backbone, transformer:transformer.DeformableTransformer, num_classes, num_queries, num_feature_levels, aux_loss=True):
        super().__init__()
        self.num_queries = num_queries
        self.num_classes = num_classes
        self.num_feature_levels = num_feature_levels
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = layers.LocFFN(hidden_dim, hidden_dim, 4, 3)
        if num_feature_levels > 1:
            num_backbone_outs = 4
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(16, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(16, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        self.backbone = backbone
        self.transformer = transformer
        self.query = nn.Embedding(num_queries, hidden_dim)
        self.pos_query = nn.Embedding(num_queries, hidden_dim)
        num_pred = transformer.decoder.num_blocks + 1
        self.aux_loss= aux_loss
        if aux_loss:
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
    
    def forward(self, sample):
        features, pos = self.backbone(sample)

        srcs = []
        for layer_num, feat in enumerate(features):
            srcs.append(self.input_proj[layer_num](feat))

        t_feats, init_reference, inter_references = self.transformer(srcs, pos, self.query, self.pos_query)
        outputs_classes = []
        outputs_coords = []
        for lvl in range(t_feats.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = layers.inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](t_feats[lvl]).squeeze(-1)
            tmp = self.bbox_embed[lvl](t_feats[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)
        #preds
        #logits = self.class_embed(t_feats).squeeze(-1)
        #bboxes = self.bbox_embed(t_feats)
        out = {'bbox': outputs_coord[-1], 'logits': outputs_class[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        return out
    
    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        return [{'logits':a, 'bbox':b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]
    

class DefEncoderDet(nn.Module):
    
    def __init__(self, backbone, num_queries, num_feature_levels, emb_dim:int=256, num_blocks:int=1, stochastic_depth_rate:float=0.1):
        super().__init__()
        self.num_queries = num_queries
        self.num_feature_levels = num_feature_levels
        hidden_dim = emb_dim
        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, emb_dim))
        if num_feature_levels > 1:
            num_backbone_outs = 4
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(16, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(16, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        self.backbone = backbone
        self.transformer = transformer.DeformableEncoder(emb_dim, num_blocks, stochastic_depth_rate)
        self.norm = nn.LayerNorm(emb_dim)
        self.seq_pool = layers.SeqPool(emb_dim)
        
        #loc head
        self.cx = nn.Linear(emb_dim, num_queries)
        self.cy = nn.Linear(emb_dim, num_queries)
        self.w = nn.Linear(emb_dim, num_queries)
        self.h = nn.Linear(emb_dim, num_queries)
        self.cls_head = nn.Linear(hidden_dim, num_queries)

    def prepare_input_features(self, srcs, pos_embeds):
        src_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, pos_embed) in enumerate(zip(srcs, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
        src_flatten = torch.cat(src_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        return src_flatten, lvl_pos_embed_flatten, spatial_shapes, level_start_index

    def forward(self, sample):
        features, pos = self.backbone(sample)

        srcs = []
        for layer_num, feat in enumerate(features):
            srcs.append(self.input_proj[layer_num](feat))
        src_flatten, lvl_pos_embed_flatten, spatial_shapes, level_start_index = self.prepare_input_features(srcs, pos)
        t_feats, _ = self.transformer(src_flatten, spatial_shapes, level_start_index, None)

        x = self.norm(t_feats)
        x = self.seq_pool(x)
        
        cx = self.cx(x)
        cy = self.cy(x)
        w = self.w(x)
        h = self.h(x)
        bboxes = torch.stack([cx,cy,w,h])
        bboxes = bboxes.permute(1,2,0)
        bboxes = bboxes.sigmoid()

        logits = self.cls_head(x)
        return {'bbox': bboxes, 'logits': logits} 
    

class DefEncoderCls(nn.Module):
    
    def __init__(self, backbone, num_cls:int=1, num_feature_levels:int=4, emb_dim:int=256, num_blocks:int=1, stochastic_depth_rate:float=0.1):
        super().__init__()
        self.num_feature_levels = num_feature_levels
        hidden_dim = emb_dim
        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, emb_dim))
        if num_feature_levels > 1:
            num_backbone_outs = 4
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(16, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(16, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        self.backbone = backbone
        self.transformer = transformer.DeformableEncoder(emb_dim, num_blocks, stochastic_depth_rate)
        self.norm = nn.LayerNorm(emb_dim)
        self.seq_pool = layers.SeqPool(emb_dim)
        
        #loc head
        self.cls_head = nn.Linear(hidden_dim, num_cls)

    def prepare_input_features(self, srcs, pos_embeds):
        src_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, pos_embed) in enumerate(zip(srcs, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
        src_flatten = torch.cat(src_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        return src_flatten, lvl_pos_embed_flatten, spatial_shapes, level_start_index

    def forward(self, sample):
        features, pos = self.backbone(sample)
        srcs = []
        for layer_num, feat in enumerate(features):
            srcs.append(self.input_proj[layer_num](feat))
        
        src_flatten, lvl_pos_embed_flatten, spatial_shapes, level_start_index = self.prepare_input_features(srcs, pos)
        #print(f'src_flatten:{src_flatten}')
        #print(f'\nlvl_pos_embed_flatten:{lvl_pos_embed_flatten}')
        #print(f'\nspatial_shapes:{spatial_shapes}')
        #print(f'\nlevel_start_index:{level_start_index}')
        t_feats, _ = self.transformer(src_flatten, spatial_shapes, level_start_index, None)

        x = self.norm(t_feats)
        x = self.seq_pool(x)
        

        logits = self.cls_head(x)
        return logits 



def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])