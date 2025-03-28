from peft import (
    LoraConfig,
    PeftModel,
    LoraModel,
    PeftModelForCausalLM,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from peft.peft_model import PEFT_TYPE_TO_MODEL_MAPPING
from peft.utils import _set_trainable  

import torch
from transformers import LlamaForCausalLM
try:
    from transformers import MistralForCausalLM as Mist
except ImportError:
    Mist = None
from omegaconf import DictConfig
import hydra
import deepspeed

def peft_model_animegamer(
    model, 
    peft_config=None, 
    model_id=None, 
    vocab_size=None, 
    torch_dtype='bf16'
):
    # Determine the appropriate torch data type based on the input string.
    if torch_dtype in ['bf16', 'bfloat16']:
        torch_dtype = torch.bfloat16
    elif torch_dtype in ['fp16', 'float16']:
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    # If the model is provided as a configuration dictionary, instantiate it using Hydra.
    if isinstance(model, DictConfig):
        model = hydra.utils.instantiate(model, torch_dtype=torch_dtype)

    # Ensure that exactly one of peft_config or model_id is provided.
    assert (peft_config is None) + (model_id is None) == 1

    # If a new vocabulary size is specified, update the model's token embeddings accordingly.
    if vocab_size is not None:
        old_vocab_size = model.config.vocab_size
        if old_vocab_size != vocab_size:
            print(f'old vocab size: {old_vocab_size}, new vocab size: {vocab_size}')

        print(f'Length of tokenizer and resize embedding: {vocab_size}')
        model.resize_token_embeddings(vocab_size)

        # If the new vocabulary size is larger, initialize the new embeddings.
        if vocab_size - old_vocab_size > 0:
            # Use DeepSpeed's utility for handling parameters across multiple GPUs.
            with deepspeed.zero.GatheredParameters(
                list(model.get_input_embeddings().parameters()) + 
                list(model.get_output_embeddings().parameters()), 
                modifier_rank=0
            ):
                # Get the current input embeddings.
                input_embeddings = model.get_input_embeddings().weight.data
                # Calculate the average of the existing embeddings to initialize new ones.
                input_embeddings_avg = input_embeddings[:-vocab_size + old_vocab_size].mean(
                    dim=0, keepdim=True
                )
                # Set the new embeddings to the calculated average.
                input_embeddings[-vocab_size + old_vocab_size:] = input_embeddings_avg
                
                # If output embeddings are separate and not tied to input embeddings, initialize them similarly.
                if model.get_output_embeddings() is not None and not model.config.tie_word_embeddings:
                    output_embeddings = model.get_output_embeddings().weight.data
                    output_embeddings_avg = output_embeddings[:-vocab_size + old_vocab_size].mean(
                        dim=0, keepdim=True
                    ) * 3
                    output_embeddings[-vocab_size + old_vocab_size:] = output_embeddings_avg


    if peft_config is not None:
        print('peft config: ', peft_config)
        peft_model = get_peft_model(model=model, peft_config=peft_config)
        peft_model.get_input_embeddings().requires_grad_(True)
        peft_model.get_output_embeddings().requires_grad_(True)

        peft_model.print_trainable_parameters()

    else:
        peft_model = PeftModel.from_pretrained(model=model, model_id=model_id)

    return peft_model