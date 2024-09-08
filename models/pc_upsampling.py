import torch
import torch.nn as nn
from torch_cluster import knn_graph
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from .DynamicEdgeConv import DynamicEdgeConv
from pytorch3d.loss import chamfer_distance

class MLP_CONV(nn.Module):
    def __init__(self, in_channel, layer_dims, bn=None):
        super(MLP_CONV, self).__init__()
        layers = []
        last_channel = in_channel
        for out_channel in layer_dims[:-1]:
            layers.append(nn.Conv1d(last_channel, out_channel, 1))
            if bn:
                layers.append(nn.BatchNorm1d(out_channel))
            layers.append(nn.ReLU())
            last_channel = out_channel
        layers.append(nn.Conv1d(last_channel, layer_dims[-1], 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.mlp(inputs)

class MLP_Res(nn.Module):
    def __init__(self, in_dim=128, hidden_dim=None, out_dim=128):
        super(MLP_Res, self).__init__()
        if hidden_dim is None:
            hidden_dim = in_dim
        self.conv_1 = nn.Conv1d(in_dim, hidden_dim, 1)
        self.conv_2 = nn.Conv1d(hidden_dim, out_dim, 1)
        self.conv_shortcut = nn.Conv1d(in_dim, out_dim, 1)

    def forward(self, x):
        """
        Args:
            x: (B, out_dim, n)
        """
        shortcut = self.conv_shortcut(x)
        out = self.conv_2(torch.relu(self.conv_1(x))) + shortcut
        return out
    
class Upsample_Reshape_Unit(nn.Module):
    def __init__(self, params):
        super(Upsample_Reshape_Unit, self).__init__()
        self.params = params

    def forward(self, points, point_features):
        device = point_features.device

        # point shape: (#point, 8)
        # point_feature shape: (#point, C)

        # output_feature shape: (B, C, lidar_points/scale)
        # output_point shape: (B, 3, lidar_points/scale)
        scale = 2**self.params.num_upsample_unit
        output_feature = torch.autograd.Variable(torch.zeros((points.batch.max()+1, point_features.shape[-1],\
                                                              self.params.lidar_points//scale)), requires_grad=True).to(device)
        output_point = torch.autograd.Variable(torch.zeros((points.batch.max()+1, 3, \
                                                            self.params.lidar_points//scale)), requires_grad=True).to(device)

        for i in range(points.batch.max()+1):
            mask = points.batch == i
            feature = point_features[mask].unsqueeze(0).permute(0, 2, 1)
            point = points.x[:, :3][mask].unsqueeze(0).permute(0, 2, 1)
            num_point = point.shape[-1]

            up_ratio = self.params.lidar_points/num_point/scale
            duplicate_point = nn.Upsample(scale_factor=up_ratio)(point)
            duplicate_feature = nn.Upsample(scale_factor=up_ratio)(feature)
            
            if duplicate_feature.shape[-1] != self.params.lidar_points//scale:
                up_ratio = self.params.lidar_points/duplicate_point.shape[-1]/scale
                duplicate_point = nn.Upsample(scale_factor=up_ratio)(duplicate_point)
                duplicate_feature = nn.Upsample(scale_factor=up_ratio)(duplicate_feature)

            output_feature.data[i] = duplicate_feature
            output_point.data[i] = duplicate_point

        return output_point, output_feature

class Upsampling_unit(nn.Module):
    """
    Point upsampling unit
    
    Input:
        point_feat: input feature, (B, dim_feat, N_input)
        points: input points, (B, 3, N_input)
    Output:
        up_feat: upsampled feature, (B, dim, up_ratio * N_input)
        duplicated_point: upsampled results, (B, 3, up_ratio * N_input)
    """
    def __init__(self, params, up_ratio=2):
        super(Upsampling_unit, self).__init__()
        self.mlp_1 = MLP_CONV(in_channel=params.radar_gcn_channel_out, layer_dims=[64, 32])
        self.mlp_2 = MLP_Res(in_dim=params.radar_gcn_channel_out*2, hidden_dim=128, out_dim=params.radar_gcn_channel_out)
        self.deconv_branch = nn.ConvTranspose1d(32, params.radar_gcn_channel_out, up_ratio, up_ratio, bias=False) 
        self.duplicated_branch = nn.Upsample(scale_factor=up_ratio)

    def forward(self, points, point_features):
        deconved_feat = self.deconv_branch(self.mlp_1(point_features)) 
        duplicated_feat = self.duplicated_branch(point_features)
        up_feat = self.mlp_2(torch.cat([deconved_feat, duplicated_feat], 1))
        up_feat = torch.relu(up_feat)
        duplicated_point = self.duplicated_branch(points)
        return duplicated_point, up_feat
    
class Upsample_GNN(nn.Module):
    def __init__(self, params):
        super(Upsample_GNN, self).__init__()
        self.params = params
        in_channels = params.radar_gcn_channel_out
        heads = params.gat_head
        self.gat1 = GATConv(in_channels=in_channels+3, out_channels=in_channels//heads, heads=heads)
        self.act = nn.ELU(inplace=True)
        self.gat2 = GATConv(in_channels=in_channels, out_channels=in_channels//heads, heads=heads)
        self.mlp = nn.Linear(in_features=2*in_channels, out_features=in_channels)


    def forward(self, point_features, point):
        device = point_features.device
        # create dynamic graph
        # B: batch size
        # F: feature size
        # N: number of points
        B, NF, N = point_features.shape
        point_features = torch.cat([point_features, point], 1)
        point_features = point_features.permute(0, 2, 1).reshape((-1, NF+3)) # (B*N, NF+3)
        batch = torch.arange(B, device=device).repeat_interleave(N)
        edge_index = knn_graph(point_features, batch=batch, k=self.params.k)
        point_features1 = self.act(self.gat1(point_features, edge_index))
        point_features1 = F.dropout(point_features1, p=0.5, training=self.training)

        edge_index = knn_graph(point_features1, batch=batch, k=self.params.k)
        point_features2 = self.act(self.gat2(point_features1, edge_index))
        point_features2 = F.dropout(point_features2, p=0.5, training=self.training)

        point_features2 = torch.cat((point_features1, point_features2), 1)
        point_features2 = self.act(self.mlp(point_features2))

        point_features2 = point_features2.reshape((B, N, NF)).permute(0, 2, 1)
        return point_features2


class Upsample_branch(nn.Module):
    def __init__(self, params):
        super(Upsample_branch, self).__init__()
        self.params = params
        self.upsample_reshape = Upsample_Reshape_Unit(params)
        # self.upsample_unit = Upsampling_unit(params)
        self.upsample_unit = nn.ModuleList([Upsampling_unit(params) for _ in range(self.params.num_upsample_unit)])

        self.regressor = MLP_CONV(in_channel=params.radar_gcn_channel_out, layer_dims=[params.lidar_channel_out])
        self.last_conv = nn.Conv1d(params.lidar_channel_out, 3, 1)

    def forward(self, points, point_features):
        duplicated_point, up_feat = self.upsample_reshape(points, point_features)

        # duplicated_point, up_feat= self.upsample_unit(duplicated_point, up_feat)

        for u in self.upsample_unit:
            duplicated_point, up_feat= u(duplicated_point, up_feat)

        lidar_features = self.regressor(up_feat)
        offset = self.last_conv(lidar_features)

        # up_point = duplicated_point + torch.tanh(offset)
        up_point = duplicated_point + offset



        return up_point, lidar_features