import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadCrossAttention(nn.Module):
    def __init__(self, params, feature_dim1, feature_dim2, hidden_dim, num_heads):
        super(MultiHeadCrossAttention, self).__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim

        assert hidden_dim % num_heads == 0, 'Hidden dimension must be divisible by the number of heads'
        self.head_dim = hidden_dim // num_heads

        self.query_transform = nn.Linear(feature_dim1, hidden_dim)
        self.key_transform = nn.Linear(feature_dim2, hidden_dim)
        self.value_transform = nn.Linear(feature_dim2, hidden_dim)

        # output linear layer
        self.out_transform = nn.Linear(hidden_dim, hidden_dim)
        
        # scaling factor
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim]))
    
    def forward(self, tensor1, tensor2):
        device = tensor1.device
        scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)
        batch_size = tensor1.shape[0]

        # transform queries, keys, values
        queries = self.query_transform(tensor1).view(batch_size, self.num_heads, self.head_dim)
        keys = self.key_transform(tensor2).view(batch_size, self.num_heads, self.head_dim)
        values = self.value_transform(tensor2).view(batch_size, self.num_heads, self.head_dim)

        # compute scaled dot product attention for each head
        # attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / self.scale
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / scale

        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_output = torch.matmul(attention_weights, values)

        # concatenate the output of all heads
        concatenated = attention_output.view(batch_size, self.hidden_dim)

        # final linear transformation
        output = self.out_transform(concatenated)

        return output