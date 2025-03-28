import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from timm.models.layers import DropPath
from fairscale.nn.checkpoint.checkpoint_activations import checkpoint_wrapper

class Mlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        # Set output features to input features if not provided
        out_features = out_features or in_features
        # Set hidden features to input features if not provided
        hidden_features = hidden_features or in_features
        # First fully connected layer
        self.fc1 = nn.Linear(in_features, hidden_features)
        # Activation layer
        self.act = act_layer()
        # Second fully connected layer
        self.fc2 = nn.Linear(hidden_features, out_features)
        # Dropout layer
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        # Apply first fully connected layer
        x = self.fc1(x)
        # Apply activation function
        x = self.act(x)
        # Apply dropout
        x = self.drop(x)
        # Apply second fully connected layer
        x = self.fc2(x)
        # Apply dropout
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        # Calculate the dimension of each head
        head_dim = dim // num_heads
        # Scale factor for attention scores
        self.scale = qk_scale or head_dim**-0.5
        # Linear layer to generate query, key, and value
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        # Dropout layer for attention scores
        self.attn_drop = nn.Dropout(attn_drop)
        # Linear layer for output projection
        self.proj = nn.Linear(dim, dim)
        # Dropout layer for output projection
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_gradients = None
        self.attention_map = None

    def save_attn_gradients(self, attn_gradients):
        # Save attention gradients
        self.attn_gradients = attn_gradients

    def get_attn_gradients(self):
        # Get attention gradients
        return self.attn_gradients

    def save_attention_map(self, attention_map):
        # Save attention map
        self.attention_map = attention_map

    def get_attention_map(self):
        # Get attention map
        return self.attention_map

    def forward(self, x, mask=None, register_hook=False):
        B, N, C = x.shape
        # Generate query, key, and value
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]  # Split query, key, and value

        # Calculate attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            mask = mask.repeat(1, attn.shape[1], attn.shape[2], 1)
            attn = attn + mask
        # Apply softmax to get attention probabilities
        attn = attn.softmax(dim=-1)
        # Apply dropout to attention probabilities
        attn = self.attn_drop(attn)

        if register_hook:
            self.save_attention_map(attn)
            attn.register_hook(self.save_attn_gradients)

        # Apply attention to value
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        # Apply output projection
        x = self.proj(x)
        # Apply dropout to output projection
        x = self.proj_drop(x)
        return x

class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        use_grad_checkpointing=False,
    ):
        super().__init__()
        # Normalization layer before attention
        self.norm1 = norm_layer(dim)
        # Attention layer
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        # Drop path for stochastic depth
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        # Normalization layer before MLP
        self.norm2 = norm_layer(dim)
        # Hidden dimension for MLP
        mlp_hidden_dim = int(dim * mlp_ratio)
        # MLP layer
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        if use_grad_checkpointing:
            # Use gradient checkpointing for attention and MLP layers
            self.attn = checkpoint_wrapper(self.attn)
            self.mlp = checkpoint_wrapper(self.mlp)


    def forward(self, c, mask=None, register_hook=False):
        # Apply attention layer with residual connection
        c = c + self.drop_path(self.attn(self.norm1(c), mask=mask, register_hook=register_hook))
        # Apply MLP layer with residual connection
        c = c + self.drop_path(self.mlp(self.norm2(c)))
        return c