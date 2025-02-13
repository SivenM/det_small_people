import torch 
import torch.nn as nn
import timm
from models import layers
from models import pos_emb_layers


class ConvNextBackbone(nn.Module):

    def __init__(self, pretrained:bool=True, features_only:bool=True):
        super().__init__()
        self.conv_next = timm.create_model('convnextv2_nano',
                                           pretrained=pretrained,
                                           features_only=features_only,
                                           )
        #self.conv_next.eval()
        self.num_channels = [80,160,320,640]
        self.pos_embs = nn.ModuleList([
                        pos_emb_layers.PositionEmbeddingLearned(),
                        pos_emb_layers.PositionEmbeddingLearned(),
                        pos_emb_layers.PositionEmbeddingLearned(),
                        pos_emb_layers.PositionEmbeddingLearned(),
                        ])
    def forward(self, x):
        features = self.conv_next(x)
        pos_list = []
        for i, feat in enumerate(features):
            pos_list.append(
                self.pos_embs[i](feat)
            )
        return features, pos_list
    

class DDBackbone(nn.Module):
    def __init__(self, C=1, layers_dim=[32,64,128,256,512,1024]):
        super().__init__()
        self.C = C
        self.tokenizator = layers.DetrTokenizator(C, layers_dim)
        self.pos_embs = nn.ModuleList([
                        pos_emb_layers.PositionEmbeddingLearned(),
                        pos_emb_layers.PositionEmbeddingLearned(),
                        pos_emb_layers.PositionEmbeddingLearned(),
                        pos_emb_layers.PositionEmbeddingLearned(),
                        ])
        self.num_channels = [128,256,512,1024]
        
    def forward(self, x):
        assert self.C == x.shape[1]
        features = self.tokenizator(x)
        pos_list = []
        for i, feat in enumerate(features):
            pos_list.append(
                self.pos_embs[i](feat)
            )
        return features, pos_list
        