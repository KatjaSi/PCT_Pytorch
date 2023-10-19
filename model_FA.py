import torch
import torch.nn as nn
import torch.nn.functional as F

from util import sample_and_group
from data import load_data
from flash_attn import flash_attn_qkvpacked_func, flash_attn_func

class SPCT_FA(nn.Module):


    def __init__(self, output_channels=40):
        super(SPCT_FA, self).__init__()

        self.conv1 = nn.Conv1d(3, 128, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(128, 256, kernel_size=1, bias=False) #128 <-> 256

        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(256)

        self.encoder1 = EncoderModule(256, num_heads=4)
        self.encoder2 = EncoderModule(256, num_heads=4)
        self.encoder3 = EncoderModule(256, num_heads=4)
        self.encoder4 = EncoderModule(256, num_heads=4)

        self.conv_fuse = nn.Sequential(nn.Conv1d(1024, 1024, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(1024), # 1024
                                   nn.LeakyReLU(negative_slope=0.2))
        
        self.linear1 = nn.Linear(1024, 512, bias=False) #1024, 512
        self.bn6 = nn.BatchNorm1d(512) # 512
        self.dp1 = nn.Dropout(0.5)
 
        self.linear2 = nn.Linear(512, 256) #512, 256
        self.bn7 = nn.BatchNorm1d(256) #256
        self.dp2 = nn.Dropout(0.5)

        self.linear3 = nn.Linear(256, output_channels) #256

    def forward(self, x):
        """
        Forward pass of the NPCT model.

        This method processes the input point cloud data through the NPCT architecture and produces
        class predictions.

        Args:
            x (torch.Tensor): Input point cloud data with shape (batch_size, num_features, num_points).

        Returns:
            torch.Tensor: Predicted class scores for each input point cloud in the batch.
                          Shape: (batch_size, num_output_classes).
        """
        batch_size, _, _ = x.size()
        x = F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.2) #  x = F.relu(self.bn1(self.conv1(x)))
        x = F.leaky_relu(self.bn2(self.conv2(x)), negative_slope=0.2)   #  x = F.relu(self.bn1(self.conv1(x)))

        x1 = self.encoder1(x)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        x4 = self.encoder4(x3)
        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.conv_fuse(x)
        x = F.adaptive_max_pool1d(x, 1)
        x = x.view(batch_size, -1)

        x = self.linear1(x)
        x = self.bn6(x) 
        x = F.leaky_relu(x, negative_slope=0.2) 
        x = self.dp1(x) 

        x = self.linear2(x)
        #x = self.bn7(x)
        x = F.leaky_relu(x, negative_slope=0.2) # F.relu(x)
        x = self.dp2(x) 

        x = self.linear3(x)
        return x


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, in_features, head_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        
        self.in_features = in_features
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.q_conv = nn.Conv1d(in_features, in_features, 1, bias=False) # Q
        self.k_conv = nn.Conv1d(in_features, in_features, 1, bias=False) # K, queries and keys of same dimentionality
        self.q_conv.weight = self.k_conv.weight
        self.q_conv.bias = self.k_conv.bias

        self.v_conv = nn.Conv1d(in_features, in_features, 1)
        
        assert  num_heads*head_dim == in_features
        
        self.out_linear = nn.Linear(head_dim * num_heads, in_features)
        
    def split_heads(self, x):
        batch_size, seq_len, _ = x.size()
        return x.view(batch_size, seq_len, self.num_heads, self.head_dim)
    
    def forward(self, x):
        # Split heads
        q = self.q_conv(x).permute(0, 2, 1)#.unsqueeze(2) # TODO: smth else for multiple heads
        k = self.k_conv(x).permute(0, 2, 1)#.unsqueeze(2)  # K = F_in * W_k
        v = self.v_conv(x).permute(0, 2, 1)#.unsqueeze(2)  # V = F_in * W_v
       # fa = flash_attn_func(q, k, v)  
        
        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)
        
        x = flash_attn_func(q, k, v) 
        x = torch.cat(tuple(x.unbind(3)), dim=2) #  concatenate outputs from heads
        x = x.permute(0,2,1)
        return x


class EncoderModule(nn.Module):
    """
    A module implementing an encoder block.

    This module incorporates a Multi-Head Self Attention mechanism followed by linear transformations
    and batch normalization to process input data.

    Args:
        in_features (int, optional): The dimensionality of each input feature (default: 256).
        num_heads (int, optional): The number of attention heads to use in the Multi-Head Self Attention mechanism (default: 8).
    """
    def __init__(self, in_features=256, num_heads=8):
        super(EncoderModule, self).__init__()
        
        self.mh_sa = MultiHeadSelfAttention(in_features=in_features, head_dim=int(in_features/num_heads), num_heads=num_heads) # in_features is dim of each point
        self.bn_after_sa = nn.BatchNorm1d(in_features)

        # Linear layer is to learn  interactions within each embedding independently of other embeddings
        self.linear_after_sa =  nn.Linear(in_features,in_features) 
        self.bn_after_linear_sa = nn.BatchNorm1d(in_features) 
        
    def forward(self, x):
        x_attention = self.mh_sa(x)
        x = x + x_attention
        x = self.bn_after_sa(x)
        x = x.permute(0, 2, 1)
        x_linear = self.linear_after_sa(x)
        x = x + x_linear
        x = x.permute(0, 2, 1)
        x = self.bn_after_linear_sa(x)
        
        return x




class SA_Layer(nn.Module):
    """
    Offset-Self-Attention
    """
    def __init__(self, channels):
        super(SA_Layer, self).__init__()
        self.q_conv = nn.Conv1d(channels, channels , 1, bias=False) # Q
        self.k_conv = nn.Conv1d(channels, channels , 1, bias=False) # K, queries and keys of same dimentionality
        self.q_conv.weight = self.k_conv.weight
        self.q_conv.bias = self.k_conv.bias

        ##
        self.q_fc = nn.Linear(channels, channels, bias=False)  # Fully connected layer for Q
        self.k_fc = nn.Linear(channels, channels, bias=False)  # Fully connected layer for K
        self.v_fc = nn.Linear(channels, channels)
        ##

        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x_q = self.q_conv(x).permute(0, 2, 1).unsqueeze(2) # TODO: smth else for multiple heads
        x_k = self.k_conv(x).permute(0, 2, 1).unsqueeze(2)  # K = F_in * W_k
        x_v = self.v_conv(x).permute(0, 2, 1).unsqueeze(2)  # V = F_in * W_v  
        fa = flash_attn_func(x_q, x_k, x_v)     
        return fa

def main():
    test_points, test_labels = load_data("test")
    pct = SPCT_FA().half().to("cuda")
    outputs = pct(torch.from_numpy(test_points[:4]).half().to('cuda').permute(0, 2, 1))
    print(outputs)

if __name__ == '__main__':
    main()