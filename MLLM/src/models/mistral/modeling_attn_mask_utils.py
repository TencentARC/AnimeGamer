# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch


@dataclass
class AttentionMaskConverter:
    """
    A utility attention mask class that allows one to:
        - Create a causal 4d mask
        - Create a causal 4d mask with slided window
        - Convert a 2d attention mask (batch_size, query_length) to a 4d attention mask (batch_size, 1, query_length,
          key_value_length) that can be multiplied with attention scores
    """

    is_causal: bool
    sliding_window: int

    def __init__(self, is_causal: bool, sliding_window: Optional[int] = None):
        self.is_causal = is_causal
        self.sliding_window = sliding_window

    def to_causal_4d(
        self,
        batch_size: int,
        query_length: int,
        key_value_length: int,
        dtype: torch.dtype,
        device: Union[torch.device, "str"] = "cpu",
    ) -> Optional[torch.Tensor]:

        if not self.is_causal:
            raise ValueError(f"Please use `to_causal_4d` only if {self.__class__} has `is_causal` set to True.")

        # If shape is not cached, create a new causal mask and cache it
        input_shape = (batch_size, query_length)
        past_key_values_length = key_value_length - query_length

        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        causal_4d_mask = None
        if input_shape[-1] > 1 or self.sliding_window is not None:
            causal_4d_mask = self._make_causal_mask(
                input_shape,
                dtype,
                device=device,
                past_key_values_length=past_key_values_length,
                sliding_window=self.sliding_window,
            )

        return causal_4d_mask

    def to_4d(
        self,
        attention_mask_2d: torch.Tensor,
        query_length: int,
        dtype: torch.dtype,
        key_value_length: Optional[int] = None,
    ) -> torch.Tensor:

        input_shape = (attention_mask_2d.shape[0], query_length)

        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        causal_4d_mask = None
        if (input_shape[-1] > 1 or self.sliding_window is not None) and self.is_causal:

            past_key_values_length = key_value_length - query_length
            causal_4d_mask = self._make_causal_mask(
                input_shape,
                attention_mask_2d,
                dtype,
                device=attention_mask_2d.device,
                past_key_values_length=past_key_values_length,
                sliding_window=self.sliding_window,
            )
            #print('causal_4d_mask', causal_4d_mask.shape, causal_4d_mask)
        elif self.sliding_window is not None:
            raise NotImplementedError("Sliding window is currently only implemented for causal masking")

        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        expanded_attn_mask = self._expand_mask(attention_mask_2d, dtype, tgt_len=input_shape[-1]).to(
            attention_mask_2d.device
        )
        if causal_4d_mask is not None:
            expanded_attn_mask = causal_4d_mask.masked_fill(expanded_attn_mask.bool(), torch.finfo(dtype).min)

        # expanded_attn_mask + causal_4d_mask can cause some overflow
        expanded_4d_mask = expanded_attn_mask

        return expanded_4d_mask

    @staticmethod
    def _make_causal_mask(
        input_ids_shape: torch.Size,
        attention_mask_2d: torch.Size,
        dtype: torch.dtype,
        device: torch.device,
        past_key_values_length: int = 0,
        sliding_window: Optional[int] = None,
    ):
        """
        Make causal mask used for bi-directional self-attention.
        """
        bsz, tgt_len = input_ids_shape
        mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
        mask_cond = torch.arange(mask.size(-1), device=device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)

        new_mask = mask.unsqueeze(0).repeat(bsz, 1, 1)

        for i in range(bsz):
            indx_i_list = torch.where(attention_mask_2d[i] == -1)[0]
            
            if len(indx_i_list) > 0:
                indx_i_start = indx_i_list[0].item()
                indx_i_end = indx_i_list[-1].item()
                new_mask[i, indx_i_start:indx_i_end+1, indx_i_start:indx_i_end+1] = 0

        mask = new_mask.to(dtype)

        if past_key_values_length > 0:
            mask = torch.cat(
                [
                    torch.zeros(
                        bsz, 
                        tgt_len, 
                        past_key_values_length, 
                        dtype=dtype, 
                        device=device
                    ), 
                    mask
                ], 
                dim=-1
            )

        # add lower triangular sliding window mask if necessary
        if sliding_window is not None:
            diagonal = past_key_values_length - sliding_window + 1

            context_mask = 1 - torch.triu(torch.ones_like(mask, dtype=torch.int), diagonal=diagonal)
            mask.masked_fill_(context_mask.bool(), torch.finfo(dtype).min)

        return mask[:, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)

    @staticmethod
    def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
        """
        Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
        """
        bsz, src_len = mask.size()
        tgt_len = tgt_len if tgt_len is not None else src_len

        expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

        inverted_mask = 1.0 - torch.abs(expanded_mask)

        return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)

    @staticmethod
    def _unmask_unattended(
        expanded_mask: torch.Tensor, attention_mask: torch.Tensor, unmasked_value: Union[bool, float]
    ):
        # fmt: on

        # Get the index of the first non-zero value for every sample in the batch.
        # In the above example, indices = [[2], [0], [1]]]
        tmp = torch.arange(attention_mask.shape[1], 0, -1)
        indices = torch.argmax(attention_mask.cpu() * tmp, 1, keepdim=True)

        # expanded mask will be completely unattended.
        left_masked_rows = torch.where(indices > 0)[0]

        if left_masked_rows.shape[0] == 0:
            return expanded_mask
        indices = indices[left_masked_rows]

        max_len = torch.max(indices)
        range_tensor = torch.arange(max_len).unsqueeze(0)
        range_tensor = range_tensor.repeat(indices.size(0), 1)

        range_tensor[range_tensor >= indices] = 0

        # TODO: we may drop support for 3D attention mask as the refactor from Patrick maybe dropped this case
        if expanded_mask.dim() == 4:
            num_masks = expanded_mask.shape[1]
            if num_masks == 1:
                # Broadcast [left_masked_rows, 1], [left_masked_rows, max_len]
                mask_slice = (left_masked_rows[:, None], 0, range_tensor)
            else:
                # Broadcast [left_masked_rows, 1, 1], [1, num_masks, 1], [left_masked_rows, 1, max_len]
                mask_slice = (
                    left_masked_rows[:, None, None],
                    torch.arange(num_masks)[None, :, None],
                    range_tensor[:, None, :],
                )
        else:
            # Broadcast [left_masked_rows, 1], [left_masked_rows, max_len]
            mask_slice = (left_masked_rows[:, None], range_tensor)

        expanded_mask[mask_slice] = unmasked_value

        return expanded_mask


def _prepare_4d_causal_attention_mask(
    attention_mask: Optional[torch.Tensor],
    input_shape: Union[torch.Size, Tuple, List],
    inputs_embeds: torch.Tensor,
    past_key_values_length: int,
    sliding_window: Optional[int] = None,
):
    attn_mask_converter = AttentionMaskConverter(is_causal=True, sliding_window=sliding_window)

    key_value_length = input_shape[-1] + past_key_values_length
    attention_mask_ori = attention_mask
    # 4d mask is passed through the layers
    if attention_mask is not None:
        attention_mask = attn_mask_converter.to_4d(
            attention_mask, input_shape[-1], key_value_length=key_value_length, dtype=inputs_embeds.dtype
        )
    else:
        attention_mask = attn_mask_converter.to_causal_4d(
            input_shape[0], input_shape[-1], key_value_length, dtype=inputs_embeds.dtype, device=inputs_embeds.device
        )
    return attention_mask


# Adapted from _prepare_4d_causal_attention_mask
def _prepare_4d_attention_sdpa(
    attention_mask: Optional[torch.Tensor],
    input_shape: Union[torch.Size, Tuple, List],
    inputs_embeds: torch.Tensor,
    past_key_values_length: int,
    sliding_window: Optional[int] = None,
):
    """
    Prepares the correct `attn_mask` argument to be used by `torch.nn.functional.scaled_dot_product_attention`.
    """
    attn_mask_converter = AttentionMaskConverter(is_causal=True, sliding_window=sliding_window)

    key_value_length = input_shape[-1] + past_key_values_length
    batch_size, query_length = input_shape
    is_tracing = torch.jit.is_tracing()

    if attention_mask is not None:
        if torch.all(attention_mask == 1):
            if is_tracing:
                pass
            elif query_length == 1:
                # For query_length == 1, causal attention and bi-directional attention are the same.
                attention_mask = None
            elif key_value_length == query_length:
                attention_mask = None
            else:
                pass
    elif query_length > 1 and key_value_length != query_length:
        # See the comment above (https://github.com/pytorch/pytorch/issues/108108).
        # Ugly: we set it to True here to dispatch in the following controlflow to `to_causal_4d`.
        attention_mask = True

    if attention_mask is None:
        expanded_4d_mask = None
    elif attention_mask is True:
        expanded_4d_mask = attn_mask_converter.to_causal_4d(
            input_shape[0], input_shape[-1], key_value_length, dtype=inputs_embeds.dtype, device=inputs_embeds.device
        )
    else:
        expanded_4d_mask = attn_mask_converter.to_4d(
            attention_mask,
            input_shape[-1],
            dtype=inputs_embeds.dtype,
            key_value_length=key_value_length,
        )

        # From PyTorch 2.1 onwards, F.scaled_dot_product_attention with the memory-efficient attention backend
        if query_length > 1:
            expanded_4d_mask = AttentionMaskConverter._unmask_unattended(
                expanded_4d_mask, attention_mask, unmasked_value=0.0
            )

    return expanded_4d_mask

