import torch
import torch.nn as nn
import torch.nn.functional as F

# the 1D convolution layers (nn.Conv1d) are performing a linear transformation on the input data (L in LBR)

class PCT(nn.Module):
    """
    Implementation ov naive PCT, without neighbour embedding

    Attributes:
        args (namespace): a configuration or settings namespace containing hyperparameters and model options.
        # TODO: add args
        output_channel: number of classes in the classification problem. 
    """
    def __init__(self, output_channels=10):
        super(PCT, self).__init__()

        self.conv1 = nn.Conv1d(3, 128, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(128, 128, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)

        self.sa1 = SA_Layer(128)
        self.sa2 = SA_Layer(128)
        self.sa3 = SA_Layer(128)
        self.sa4 = SA_Layer(128)

        self.conv_fuse = nn.Sequential(nn.Conv1d(512, 1024, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(1024),
                                   nn.LeakyReLU(negative_slope=0.2))
        
        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=0.5)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        """
        Forward pass of the PCT model.

        This method processes the input point cloud data through the PCT architecture and produces
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

        x1 = self.sa1(x)
        x2 = self.sa2(x1)
        x3 = self.sa3(x2)
        x4 = self.sa4(x3)
        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.conv_fuse(x)
        x = F.adaptive_max_pool1d(x, 1)
        x = x.view(batch_size, -1)

        x = self.linear1(x)
        x = self.bn6(x) # will not work for batch size = 1
        x = F.leaky_relu(x, negative_slope=0.2)
        x = self.dp1(x)
        x = self.linear2(x)
        x = self.bn7(x)
        x = F.leaky_relu(x, negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x


class SA_Layer(nn.Module):
    def __init__(self, channels):
        super(SA_Layer, self).__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.q_conv.bias = self.k_conv.bias

        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x_q = self.q_conv(x).permute(0, 2, 1)
        x_k = self.k_conv(x)
        x_v = self.v_conv(x)

        energy = torch.bmm(x_q, x_k)

        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdim=True))
        # b, c, n
        x_r = torch.bmm(x_v, attention)
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r
        return x

def main():
    pct = PCT()
    print(str(pct))
    

if __name__ == '__main__':
    main()