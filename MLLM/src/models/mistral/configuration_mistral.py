# coding=utf-8
# Copyright statement indicating that the copyright belongs to Mistral AI and the HuggingFace Inc. team.

""" Mistral model configuration"""
# Description of the file, indicating that it contains configuration settings for the Mistral model.

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)
# Get a logger for the current module.

class MistralConfig(PretrainedConfig):
    # Define the MistralConfig class, inheriting from PretrainedConfig.

    model_type = "mistral"
    # Define the model type as "mistral".

    keys_to_ignore_at_inference = ["past_key_values"]
    # Define keys to ignore during inference.

    def __init__(
        self,
        vocab_size=32000,  # Vocabulary size, default is 32000.
        hidden_size=4096,  # Size of the hidden layers.
        intermediate_size=14336,  # Size of the intermediate layer.
        num_hidden_layers=32,  # Number of hidden layers.
        num_attention_heads=32,  # Number of attention heads.
        num_key_value_heads=8,  # Number of key-value heads.
        hidden_act="silu",  # Activation function, default is silu.
        max_position_embeddings=4096 * 32,  # Maximum position embeddings.
        initializer_range=0.02,  # Range of the initializer.
        rms_norm_eps=1e-6,  # Epsilon value for RMS normalization.
        use_cache=True,  # Whether to use caching.
        pad_token_id=None,  # ID for the padding token.
        bos_token_id=1,  # ID for the beginning-of-sentence token.
        eos_token_id=2,  # ID for the end-of-sentence token.
        tie_word_embeddings=False,  # Whether to tie word embeddings.
        rope_theta=10000.0,  # Theta value for ROPE positional encoding.
        sliding_window=4096,  # Size of the sliding window.
        attention_dropout=0.0,  # Dropout rate for the attention layers.
        **kwargs,  # Other keyword arguments.
    ):
        # Initialize class attributes.
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.sliding_window = sliding_window

        # For backward compatibility, if num_key_value_heads is None, set it to the value of num_attention_heads.
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.attention_dropout = attention_dropout

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )  
