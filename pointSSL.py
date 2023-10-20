import torch
import torch.nn as nn
import torch.nn.functional as F

from model_FA import EncoderModule

class POINT_SSL(nn.Module):

    def __init__(self, in_channels=3):
        super(SPCT_FA, self).__init__()

        self.embedding_module = EmbeddingModule(in_channels=in_channels, out_channels=256)

        self.encoder1 = EncoderModule(256, num_heads=4)
        self.encoder2 = EncoderModule(256, num_heads=4)
        self.encoder3 = EncoderModule(256, num_heads=4)
        self.encoder4 = EncoderModule(256, num_heads=4)

        self.conv_fuse = nn.Sequential(nn.Conv1d(256, 256, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(256), 
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(256, 128, bias=False)                       
        self.bn6 = nn.BatchNorm1d(128) 
        self.dp1 = nn.Dropout(0.5)
        self.linear2 = nn.Linear(128, 128) # this output is the learned rep 
        # TODO: add CLS Token

        # projection layers
        self.linear3 = nn.Linear(128, 64) 

    def forward(self,x):
        pass
    # TODO: implement 





class EmbeddingModule(nn.Module):
    """
    A module for projecting a 3D point cloud into an embedding space.

    This module transforms the input point cloud into a higher-dimensional embedding space.

     Args:
        in_channels (int, optional): The number of input channels representing the dimensions of the input point cloud (default: 3).

    Input:
        x (torch.Tensor): The input point cloud tensor of shape (batch_size, num_points, 3).

    Output:
        torch.Tensor: The output tensor after projection, of shape (batch_size, num_points, embedding_dim).
    """

    def __init__(self, in_channels=3, out_channels=256):
        """
        Initializes an EmbeddingModule instance.

        This module consists of convolutional layers and batch normalization operations
        to project a 3D point cloud into an embedding space.
        """
        super(EmbeddingModule, self).__init__()

        # Convolutional Layers
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=128, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=1, bias=False)

        # Batch Normalization Layers
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        """
        Forward pass through the EmbeddingModule.

        Applies a series of convolutional layers and batch normalization operations to
        project the input point cloud into an embedding space.

        Args:
            x (torch.Tensor): The input point cloud tensor of shape (batch_size, num_points, 3).

        Returns:
            torch.Tensor: The output tensor after projection, of shape (batch_size, num_points, embedding_dim).
        """
        # Apply first convolution and batch normalization
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.functional.relu(x)

        # Apply second convolution and batch normalization
        x = self.conv2(x)
        x = self.bn2(x)
        x = nn.functional.relu(x)

        return x


