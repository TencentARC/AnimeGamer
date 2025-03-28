import time  # Import the time module for time-related tasks
import torch  # Import PyTorch for deep learning tasks
import torch.nn as nn  # Import neural network module from PyTorch
import pyrootutils  # Import pyrootutils for managing project roots
import torch.distributed as dist  # Import distributed computing module from PyTorch
import torch.nn.functional as F  # Import functional API from PyTorch
import math  # Import math module for mathematical operations
# Setup the project root directory and add it to the PYTHONPATH
pyrootutils.setup_root(__file__, indicator='.project-root', pythonpath=True)
from ..train.dist_utils import concat_all_gather  # Import utility function for distributed training


class ProjectionLayer(nn.Module):
    # Define a projection layer class inheriting from nn.Module
    def __init__(self, num_queries, embed_dim, input_dim, output_dim) -> None:
        super().__init__()  # Initialize the superclass (nn.Module)
        self.num_queries = num_queries  # Number of query vectors
        self.embed_dim = embed_dim  # Dimension of each embedding vector
        self.input_dim = input_dim  # Dimension of input features
        self.output_dim = output_dim  # Dimension of output features

        # Define a sequential model for projection
        self.proj = nn.Sequential(
            nn.Linear(input_dim, output_dim),  # Linear transformation from input to output dimension
            nn.GELU(),  # Gaussian Error Linear Unit (GELU) activation function
            nn.Linear(output_dim, output_dim)  # Linear transformation for further processing
        )

    def forward(self, image_embeds):
        return self.proj(image_embeds)