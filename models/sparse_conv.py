import torch.nn as nn
import torch
from scipy.signal import gaussian
import numpy as np

class SparseConv(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 dilation=1,
                 activation='relu'):
        super().__init__()

        padding1 = kernel_size//2
        padding = (1+dilation * (kernel_size-1))//2

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            bias=False)

        self.bias = nn.Parameter(
            torch.zeros(out_channels), 
            requires_grad=True)

        self.sparsity = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            bias=False)

        kernel = torch.FloatTensor(torch.ones([kernel_size, kernel_size])).unsqueeze(0).unsqueeze(0)

        self.sparsity.weight = nn.Parameter(
            data=kernel, 
            requires_grad=False)

        if activation == 'relu':
            self.act = nn.ReLU(inplace=False)
        elif activation == 'sigmoid':
            self.act = nn.Sigmoid()
        elif activation == 'elu':
            self.act = nn.ELU()

        self.max_pool = nn.MaxPool2d(
            kernel_size, 
            stride=1, 
            padding=padding1,
            dilation=1)

        

    def forward(self, x, mask):

        x = x*mask
        x = self.conv(x)
        normalizer = 1/(self.sparsity(mask)+1e-8)
        x = x * normalizer + self.bias.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        x = self.act(x)
        
        mask = self.max_pool(mask)

        return x, mask
    

class SparseConvModuleNew(nn.Module):
    def __init__(self, params):
        super(SparseConvModuleNew, self).__init__()
        # Assuming params.radar_input_channels is the desired output channels for all paths
        output_channels = params.radar_input_channels

        # Define ModuleLists with the last SparseConv in each ensuring output_channels matches
        self.sparse_conv_0_40_new = self._create_sparse_conv_sequence(params, [11, 7, 7, 5, 5, 3], output_channels)
        self.sparse_conv_40_70_new = self._create_sparse_conv_sequence(params, [11, 7, 5, 5, 3, 3], output_channels)
        self.sparse_conv_70_new = self._create_sparse_conv_sequence(params, [11, 7, 5, 3], output_channels)

    def _create_sparse_conv_sequence(self, params, lengths, output_channels):
        sequence = []
        in_channels = params.radar_input_channels
        for i, length in enumerate(lengths):
            out_channels = 16 if i < len(lengths) - 1 else output_channels
            sequence.append(SparseConv(in_channels, out_channels, length, dilation=1, activation='elu'))
            in_channels = out_channels  # Update in_channels for the next layer
        return nn.ModuleList(sequence)
    def forward(self, x):
        masks = [
            ((x[:, 0] > 0) & (x[:, 0] <= 40)).float().unsqueeze(1),
            ((x[:, 0] > 40) & (x[:, 0] <= 70)).float().unsqueeze(1),
            (x[:, 0] > 70).float().unsqueeze(1)
        ]

        features = [x] * 3  # Initialize a list with the input tensor for each range
        
        # Process each range
        for i, sparse_conv in enumerate([self.sparse_conv_0_40_new, self.sparse_conv_40_70_new, self.sparse_conv_70_new]):
            for layer in sparse_conv:
                features[i], masks[i] = layer(features[i], masks[i])

        return features[0] + features[1] + features[2]


class SparseConvOrigin(nn.Module):
    def __init__(self, params):
        super(SparseConvOrigin, self).__init__()

        self.sparse_conv = nn.ModuleList(
            [SparseConv(params.radar_input_channels, 16, 11, dilation=1, activation='elu'),
            SparseConv(16, 16, 7, dilation=1, activation='elu'),
            SparseConv(16, 16, 5, dilation=1, activation='elu'),
            SparseConv(16, 16, 5, dilation=1, activation='elu'),
            SparseConv(16, params.radar_input_channels, 3, dilation=4, activation='elu')]
        )

        
    def forward(self, x):
        mask = (x[:, 0] > 0).float().unsqueeze(1)
        
        for l in self.sparse_conv:
            x, mask = l(x, mask)

        return x