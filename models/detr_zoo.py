import torch
from torch import nn
from torch import Tensor
from torchvision.models import resnet50
from models import layers
from models import pos_emb_layers
from models import transformer

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
        bboxes = bboxes.sigmoid()

        logits = self.cls_head(x)
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
