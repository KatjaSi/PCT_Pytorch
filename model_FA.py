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

        self.sa = SA_Layer(128)

        self.conv_fuse = nn.Sequential(nn.Conv1d(128, 1024, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(1024),
                                   nn.LeakyReLU(negative_slope=0.2))
        
        self.linear1 = nn.Linear(1024, 512, bias=False)
 
        self.linear2 = nn.Linear(512, 256)

        self.linear3 = nn.Linear(256, output_channels)

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
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.sa(x)
        x = x.squeeze(2)
        x = x.permute(0,2,1)
       # x = torch.cat((x, x, x, x), dim=1)
        x = self.conv_fuse(x)
        x = F.adaptive_max_pool1d(x, 1)
        x = x.view(batch_size, -1)

        x = self.linear1(x)

        x = F.relu(x)# x = F.leaky_relu(x, negative_slope=0.2)

        x = self.linear2(x)

        x = F.relu(x)#x = F.leaky_relu(x, negative_slope=0.2)

        x = self.linear3(x)
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

        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x_q = self.q_conv(x).permute(0, 2, 1) # Q = F_in * W_q
        x_q_e = x_q.unsqueeze(2)
        x_k = self.k_conv(x) # K = F_in * W_k
        x_k_e = x_k.permute(0, 2, 1).unsqueeze(2)
        x_v = self.v_conv(x) # V = F_in * W_v
        x_v_e = x_v.permute(0,2,1).unsqueeze(2)
        fa = flash_attn_func(x_q_e, x_k_e, x_v_e)

        #energy = torch.bmm(x_q, x_k) # batch matrix multiplication

        #attention = self.softmax(energy)

        #attention = attention / (1e-9 + attention.sum(dim=1, keepdim=True))
        # b, c, n
        #x = torch.bmm(x_v, attention) #The self-attention output features F_sa are the weighted
        # sums of the value vector using the corresponding attention weights:
        
        return fa

def main():
    test_points, test_labels = load_data("test")
    pct = SPCT_FA().half().to("cuda")
    outputs = pct(torch.from_numpy(test_points[:4]).half().to('cuda').permute(0, 2, 1))
    print(outputs)

if __name__ == '__main__':
    main()