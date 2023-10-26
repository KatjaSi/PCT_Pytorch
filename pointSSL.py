import torch
import torch.nn as nn
import torch.nn.functional as F

from model_FA import EncoderModule

class POINT_SSL(nn.Module):

    def __init__(self, in_channels=3, output_channels=40):
        super(POINT_SSL, self).__init__()

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
        self.linear2 = nn.Linear(128, 128) # global rep

        
        # projection layers
        
        self.linear3 = nn.Linear(128, 64) # This representation will go into loss function

        # added for fine-tuning, do not use linear3 when fine-tuning
        self.linear4 = nn.Linear(128, output_channels)

    def forward(self, x_prime, x=None, downstream=False):

        if x is not None:
            # Pretraining mode: compute features for both x_prime and x
            x_prime_rep, x_prime_projection = self.forward_single(x_prime,downstream)
            x_rep, x_projection = self.forward_single(x, downstream)
            return x_prime_rep, x_rep, x_prime_projection, x_projection
        else:
            # Fine-tuning mode: compute features for x_prime only
            x = self.forward_single(x_prime, downstream=downstream) # TODO: works? worked when downstream=True
            return x #, x_prime_projection
        
      #  x_prime_rep, x_prime_projection = self.forward_single(x_prime)
       # x_rep, x_projection = self.forward_single(x)
       # return x_prime_rep, x_rep, x_prime_projection, x_projection

    def forward_single(self,x, downstream):
        batch_size, _, _ = x.size()

        x = self.embedding_module(x)

        x = self.encoder1(x)
        x = self.encoder2(x)
        x = self.encoder3(x)
        x = self.encoder4(x)
        x = self.conv_fuse(x)

        # collapses the spatial dimension (num_points) to 1, resulting in a tensor of shape (batch_size, num_features, 1)
        x = F.adaptive_max_pool1d(x, 1) # TODO: this is CLS token!
        x = x.view(batch_size, -1)

        x = self.linear1(x)
        x = self.bn6(x) 
        x = F.leaky_relu(x, negative_slope=0.2) 
        x = self.dp1(x) 
        x_rep = self.linear2(x)
        x = F.leaky_relu(x_rep, negative_slope=0.2) # F.relu(x)
        #x = self.dp2(x) 
        if not downstream:
            x = self.linear3(x)
            return x_rep, x # global rep, projection
        else: # if downstream
            x = self.linear4(x)
            return x
    





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
        x = F.leaky_relu(x,negative_slope=0.2)

        # Apply second convolution and batch normalization
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.leaky_relu(x,negative_slope=0.2)

        return x


