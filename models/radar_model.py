import torch
import torch.nn as nn
import torch.nn.functional as torch_nn_func
import math
import numpy as np
from torch.nn import Linear
from torch_geometric.nn import GCNConv, GCN2Conv
import torch.nn.functional as F
from .attention import MultiHeadCrossAttention
from .pc_upsampling import Upsample_branch
from .sparse_conv import *
from .DynamicEdgeConv import DynamicEdgeConv
from torch_geometric.nn import global_max_pool, global_mean_pool
from pytorch3d.loss import chamfer_distance


camera_intrinsic = torch.tensor(
    [[1266.417203046554, 0.0, 816.2670197447984, 0.0],
     [0.0, 1266.417203046554, 491.50706579294757, 0.0],
     [0.0, 0.0, 1.0, 0.0],
     [0.0, 0.0, 0.0, 1.0]])


class Attention_enhanced_DGCNN(nn.Module):
    def __init__(self, params, radar_encoder, aggr = 'max'):
        super(Attention_enhanced_DGCNN, self).__init__()
        
        inter_channel = 32
        self.dgconv1 = DynamicEdgeConv(nn.Sequential(nn.Linear(params.radar_gcn_channel_in*2, inter_channel)), params.k, aggr)
        self.atten1  = MultiHeadCrossAttention(params, inter_channel, radar_encoder.feat_out_channels[0], \
                                               inter_channel, params.num_heads)
        self.dgconv2 = DynamicEdgeConv(nn.Sequential(nn.Linear(inter_channel * 2, inter_channel*2)), params.k, aggr)
        self.atten2  = MultiHeadCrossAttention(params, inter_channel*2, radar_encoder.feat_out_channels[1], \
                                               inter_channel*2, params.num_heads)
        self.dgconv3 = DynamicEdgeConv(nn.Sequential(nn.Linear(inter_channel * 4, inter_channel*4)), params.k, aggr)
        self.atten3  = MultiHeadCrossAttention(params, inter_channel*4, radar_encoder.feat_out_channels[2], \
                                               inter_channel*4, params.num_heads)
        self.dgconv4 = DynamicEdgeConv(nn.Sequential(nn.Linear(inter_channel * 8, inter_channel*8)), params.k, aggr)
        self.atten4  = MultiHeadCrossAttention(params, inter_channel*8, radar_encoder.feat_out_channels[3], \
                                               inter_channel*8, params.num_heads)
        self.dgconv5 = DynamicEdgeConv(nn.Sequential(nn.Linear(inter_channel * 16, inter_channel*16)), params.k, aggr)
        self.atten5  = MultiHeadCrossAttention(params, inter_channel*16, radar_encoder.feat_out_channels[4], \
                                               inter_channel*16, params.num_heads)

        self.lin1= nn.Linear(inter_channel*31, params.radar_gcn_channel_out)
        self.act = nn.ELU(inplace=True)
    def forward(self, data, radar_channels):

        x, batch = data.x, data.batch
        x0 = x[:, :-2]
        radar2d = x[:, -2:]
        # pos, batch = data.pos, data.batch
        x1 = self.dgconv1(x0, batch)
        scale = 2
        point_idx = torch.div(radar2d, scale, rounding_mode='trunc')
        channel_features = radar_channels[0][batch, :, point_idx[:, 1].long(), point_idx[:, 0].long()]
        x1 = self.atten1(x1, channel_features) + x1

        x2 = self.dgconv2(x1, batch)
        scale *= 2
        point_idx = torch.div(radar2d, scale, rounding_mode='trunc')
        channel_features = radar_channels[1][batch, :, point_idx[:, 1].long(), point_idx[:, 0].long()]
        x2 = self.atten2(x2, channel_features) + x2

        x3 = self.dgconv3(x2, batch)
        scale *= 2
        point_idx = torch.div(radar2d, scale, rounding_mode='trunc')
        channel_features = radar_channels[2][batch, :, point_idx[:, 1].long(), point_idx[:, 0].long()]
        x3 = self.atten3(x3, channel_features) + x3

        x4 = self.dgconv4(x3, batch)
        scale *= 2
        point_idx = torch.div(radar2d, scale, rounding_mode='trunc')
        channel_features = radar_channels[3][batch, :, point_idx[:, 1].long(), point_idx[:, 0].long()]
        x4 = self.atten4(x4, channel_features) + x4

        x5 = self.dgconv5(x4, batch)
        scale *= 2
        point_idx = torch.div(radar2d, scale, rounding_mode='trunc')
        channel_features = radar_channels[4][batch, :, point_idx[:, 1].long(), point_idx[:, 0].long()]
        x5 = self.atten5(x5, channel_features) + x5

        out = self.act(self.lin1(torch.cat([x1, x2, x3, x4, x5], dim=1)))
        
        return out

class Radar_feature_extraction(nn.Module):
    def __init__(self, params):
        super(Radar_feature_extraction, self).__init__()
        self.params = params

        if params.sparse_conv_type == 'distance_aware_new':
            self.sparse_conv = SparseConvModuleNew(params)
        elif params.sparse_conv_type == 'origin':
            self.sparse_conv = SparseConvOrigin(params=params)
        self.radar_encoder = encoder_radar(params)

        self.radar_gcn = Attention_enhanced_DGCNN(params, self.radar_encoder)
        self.upsample = Upsample_branch(params)
        self.linear = nn.Linear(sum(self.radar_encoder.feat_out_channels), params.radar_gcn_channel_out)
        self.cross_attention_fea = nn.ModuleList(
                                    [MultiHeadCrossAttention(params, params.radar_gcn_channel_out, self.radar_encoder.feat_out_channels[0], \
                                                    self.radar_encoder.feat_out_channels[0], params.num_heads),
                                    MultiHeadCrossAttention(params, params.radar_gcn_channel_out, self.radar_encoder.feat_out_channels[1], \
                                                    self.radar_encoder.feat_out_channels[1], params.num_heads),
                                    MultiHeadCrossAttention(params, params.radar_gcn_channel_out, self.radar_encoder.feat_out_channels[2], \
                                                    self.radar_encoder.feat_out_channels[2], params.num_heads),
                                    MultiHeadCrossAttention(params, params.radar_gcn_channel_out, self.radar_encoder.feat_out_channels[3], \
                                                    self.radar_encoder.feat_out_channels[3], params.num_heads),
                                    MultiHeadCrossAttention(params, params.radar_gcn_channel_out, self.radar_encoder.feat_out_channels[4], \
                                                    self.radar_encoder.feat_out_channels[4], params.num_heads)])
        self.cross_attention = MultiHeadCrossAttention(params, params.radar_gcn_channel_out, params.radar_gcn_channel_out, \
                                                    params.radar_gcn_channel_out, params.num_heads)


    def forward(self, radar_channels, radar_points, image, K, centroid=None, furthest_distance=None):
        device = next(self.parameters()).device

        radar2d = radar_points.x[:, -2:]
        radar_channels = self.sparse_conv(radar_channels)
        radar_channels_features = self.radar_encoder(radar_channels)

        radar_points_features = self.radar_gcn(radar_points, radar_channels_features)
        
        radar_fuse_features = torch.autograd.Variable(torch.zeros((radar_points.batch.max()+1, self.params.radar_gcn_channel_out,\
                                                                   image.shape[-2], image.shape[-1])), requires_grad=True).to(self.params.device)
        
        radar_channels_features_points = []
        radar_channels_output = []
        radar_channels_sparse = []

        radar_point_fuse_features = []
        # from pytorch3d.loss import chamfer_distance

        for i, features in enumerate(radar_channels_features):
            temp = torch.autograd.Variable(torch.zeros_like(features), requires_grad=True).to(device)
            # temp_ = temp.clone()

            scale = 2**(i+1)
            # point_idx = torch.div(radar2d, scale, rounding_mode='trunc')
            # per batch
            radar_point_fuse_features_batch = []
            radar_channels_features_points_batch = []
            for j in range(radar_channels.shape[0]):

                point_idx = torch.div(radar2d[radar_points.batch==j], scale, rounding_mode='trunc')
                channel_features_batch = features[j, :, point_idx[:, 1].long(), point_idx[:, 0].long()].permute(1, 0)

                radar_channels_features_points_batch.append(channel_features_batch)
                
                channel_fea_att = self.cross_attention_fea[i](radar_points_features[radar_points.batch==j], channel_features_batch)

                radar_point_fuse_features_batch.append(channel_fea_att)
                temp.data[j, :, point_idx[:, 1].long(), point_idx[:, 0].long()] = \
                    temp.data[j, :, point_idx[:, 1].long(), point_idx[:, 0].long()] + channel_fea_att.permute(1, 0)
            
            radar_point_fuse_features_batch = torch.cat(radar_point_fuse_features_batch, axis=0)
            radar_channels_features_points_batch = torch.cat(radar_channels_features_points_batch, axis=0)

            radar_point_fuse_features.append(radar_point_fuse_features_batch)
            radar_channels_features_points.append(radar_channels_features_points_batch)
            radar_channels_sparse.append(temp)
            features = features + temp
            radar_channels_output.append(features)

        radar_point_fuse_features = torch.cat(radar_point_fuse_features, axis=-1)
        radar_channels_features_points = torch.cat(radar_channels_features_points, axis=-1)
        radar_channels_features_points = self.linear(radar_channels_features_points)

        attention_features = self.cross_attention(radar_points_features, radar_channels_features_points)

        up_point, lidar_features = self.upsample(radar_points, attention_features)


        B, _, N = up_point.shape

        if self.params.norm_point:
            up = (up_point.permute((0, 2, 1)).clone() * furthest_distance + centroid) # (B, N, 3)
        else:
            up = up_point.permute((0, 2, 1)).clone()

        lidar_depth = up.reshape(-1, 3)[:, 2:3]

        for i in range(radar_channels.shape[0]):
            up[i] = torch.matmul(K[i], torch.cat((up[i], torch.ones(N, 1, device=device)), dim=1).T)[:3, :].T # (N, 3)

        up = up.reshape(-1, 3)
        up = up / up[:, -1:]

        lidar_feat = lidar_features.permute(0, 2, 1).reshape(-1, lidar_features.shape[1]) #(B*N, C)
        batch_idx = torch.arange(B, device=device).repeat_interleave(N).unsqueeze(1)

        valid_x = (up[:, 0]>1) & (up[:, 0]<image.shape[-1]-1)
        valid_y = (up[:, 1]>1) & (up[:, 1]<image.shape[-2]-1)
        mask = valid_x & valid_y

        up = up[mask]
        lidar_feat = lidar_feat[mask]
        batch_idx = batch_idx[mask]
        lidar_depth = lidar_depth[mask]            
        
        radar_fuse_features.data[batch_idx[:, 0], :, up[:, 1].long(), up[:, 0].long()] = \
                radar_fuse_features.data[batch_idx[:, 0], :, up[:, 1].long(), up[:, 0].long()] + lidar_feat

        radar_fuse_features.data[radar_points.batch, :, radar2d[:, 1].long(), radar2d[:, 0].long()] = \
                radar_fuse_features.data[radar_points.batch, :, radar2d[:, 1].long(), radar2d[:, 0].long()] + attention_features

        return radar_channels_output, radar_channels_sparse, radar_fuse_features, up_point


class encoder_radar(nn.Module):
    def __init__(self, params):
        super(encoder_radar, self).__init__()

        self.params = params
        import torchvision.models as models
        self.conv = torch.nn.Sequential(nn.Conv2d(params.radar_input_channels, 3, 3, 1, 1, bias=False),
                                        nn.ELU())
        if params.encoder_radar == 'resnet34':
            self.base_model_radar = models.resnet34(pretrained=True)
            self.feat_names = ['relu', 'layer1', 'layer2', 'layer3', 'layer4']
            self.feat_out_channels = [64, 64, 128, 256, 512]
        elif params.encoder_radar == 'resnet18':
            self.base_model_radar = models.resnet18(pretrained=True)
            self.feat_names = ['relu', 'layer1', 'layer2', 'layer3', 'layer4']
            self.feat_out_channels = [64, 64, 128, 256, 512]
        else:
            print('Not supported encoder: {}'.format(params.encoder))
    def forward(self, x):
        feature = x
        feature = self.conv(feature)
        skip_feat = []
        i = 1
        for k, v in self.base_model_radar._modules.items():
            if 'fc' in k or 'avgpool' in k:
                continue
            feature = v(feature)
            if any(x in k for x in self.feat_names):
                skip_feat.append(feature)
            i = i + 1
        return skip_feat

