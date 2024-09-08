import torch
import torch.nn as nn
from .image_model import encoder, bts_fuse
from .radar_model import Radar_feature_extraction
from pytorch3d.loss import chamfer_distance


def weights_init_xavier(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

class GET_UP(nn.Module):
    def __init__(self, params):
        super(GET_UP, self).__init__()
        self.params = params
        self.image_encoder = encoder(params)
        self.radar_encoder = Radar_feature_extraction(params)
        self.decoder = bts_fuse(params, self.image_encoder.feat_out_channels, self.radar_encoder.radar_encoder.feat_out_channels)

    def forward(self, image, radar_channels, radar_points, focal, K, centroid=None, furthest_distance=None):
        image_output = self.image_encoder(image)

        radar_channels_output, radar_channels_sparse, radar_fuse_features, point_up = self.radar_encoder(radar_channels, radar_points, image, K,\
                                                                                                        centroid, furthest_distance)
        
        depth_8x8_scaled, depth_4x4_scaled, depth_2x2_scaled, reduc1x1, final_depth = self.decoder(image_output, radar_channels_output, \
                                                                                                   radar_fuse_features, focal)
        
        return depth_8x8_scaled, depth_4x4_scaled, depth_2x2_scaled, reduc1x1, final_depth, point_up  
