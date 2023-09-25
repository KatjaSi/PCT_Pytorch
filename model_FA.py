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
        self.conv2 = nn.Conv1d(128, 128, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)

        #self.sa = SA_Layer(128)
        self.mh_sa1 = MultiHeadSelfAttention(in_features=128, head_dim=32, num_heads=4) # in_features is dim of each point
        self.bn_after_sa1 = nn.BatchNorm1d(128)
        self.mh_sa2 = MultiHeadSelfAttention(in_features=128, head_dim=32, num_heads=4)
        self.bn_after_sa2 = nn.BatchNorm1d(128)
        self.mh_sa3 = MultiHeadSelfAttention(in_features=128, head_dim=32, num_heads=4) # in_features is dim of each point
        self.bn_after_sa3 = nn.BatchNorm1d(128)
        self.mh_sa4 = MultiHeadSelfAttention(in_features=128, head_dim=32, num_heads=4)
        self.bn_after_sa4 = nn.BatchNorm1d(128)

        self.conv_fuse = nn.Sequential(nn.Conv1d(128, 1024, kernel_size=1, bias=False), # TODO:  128, 1024
                                   nn.BatchNorm1d(1024), # 1024
                                   nn.LeakyReLU(negative_slope=0.2))
        
        self.linear1 = nn.Linear(1024, 512, bias=False) #1024, 512
        self.bn6 = nn.BatchNorm1d(512) # 512
 
        self.linear2 = nn.Linear(512, 256) #512, 256
        self.bn7 = nn.BatchNorm1d(256) #256

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
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))    
        x1 = self.mh_sa1(x)
        x = x + x1 # add
        x = self.bn_after_sa1(x) # norm
        x2 = self.mh_sa2(x)
        x = x + x2 # add
        x = self.bn_after_sa2(x) # norm

        x3 = self.mh_sa3(x)
        x = x + x3 # add
        x = self.bn_after_sa3(x) # norm
        x4 = self.mh_sa4(x)
        x = x + x4 # add
        x = self.bn_after_sa4(x) # norm
        
      #  x2 = self.mh_sa2(x)
       # x = torch.cat((x1, x2), dim=1)
        x = self.conv_fuse(x)
        x = F.adaptive_max_pool1d(x, 1)
        x = x.view(batch_size, -1)

        x = self.linear1(x)
        x = self.bn6(x) #TODO: comment this one?

        x = F.relu(x)# x = F.leaky_relu(x, negative_slope=0.2)

        x = self.linear2(x)
        # x = self.bn7(x) 

        x = F.relu(x)#x = F.leaky_relu(x, negative_slope=0.2)

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