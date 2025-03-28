import torch
import math
import numpy as np
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from transformers import LogitsProcessor, LogitsProcessorList
from .generation import AutoImageTokenGenerationProcessor, AutoVideoTokenGenerationProcessor
from ..models.vit import Block
from torch.distributions import Normal
from functools import partial

BOI_TOKEN = '<frame>'
EOI_TOKEN = '</frame>'
FRAME_TOKEN = '<frame_{:05d}>'


def cosine_loss(rec, target):
    # Normalize the target tensor along the last dimension.
    target = target / target.norm(dim=-1, keepdim=True)
    # Normalize the reconstructed (rec) tensor along the last dimension.
    rec = rec / rec.norm(dim=-1, keepdim=True)
    # Compute the cosine similarity and then the cosine loss.
    # The cosine similarity is calculated by taking the dot product of the normalized vectors.
    # The loss is 1 minus the average of these cosine similarities.
    rec_loss = (1 - (target * rec).sum(-1)).mean()
    # Return the mean cosine loss.
    return rec_loss

def gmm_check_params(means, variances, weights):
    # Ensure that the 'means' tensor has four dimensions.
    # The expected shape is (batch_size, num_tokens, k, d), where k is the number of components,
    # and d is the dimensionality of each component.
    assert (
        means.ndim == 4
    ), "means should have shape (batch_size, num_tokens, k, d), got {}".format(
        means.shape
    )
    # Ensure that the 'means' and 'variances' tensors have the same shape.
    assert (
        means.shape == variances.shape
    ), "means and variances should have the same shape, got {} and {}".format(
        means.shape, variances.shape
    )
    # Ensure that the first three dimensions of 'means' match those of 'weights'.
    # This checks that the batch size, number of tokens, and number of components (k) are consistent.
    assert means.shape[:3] == weights.shape

def gmm_split_params(gmm_params, k, d, var_scale=1.0):
    # Extract dimensions from the input parameter tensor.
    batch_size, num_tokens, _ = gmm_params.shape

    # Split and reshape the parameters to separate means, variances, and weights.
    # Means are taken from the first k*d elements in the last dimension, then reshaped.
    means = gmm_params[..., : k * d].reshape(batch_size, num_tokens, k, d)

    # Variances are taken from the next k*d elements, reshaped similarly, and processed.
    # Softplus is applied to ensure positivity, clamped to avoid values too close to zero,
    # and scaled by 'var_scale'.
    variances = gmm_params[..., k * d : 2 * k * d].reshape(batch_size, num_tokens, k, d)
    variances = torch.clamp(F.softplus(variances), min=1e-5)
    variances = variances * var_scale

    # Weights are taken from the remaining elements in the tensor.
    weights = gmm_params[..., 2 * k * d :]

    # Return the split and processed GMM parameters.
    return means, variances, weights

def gmm_predict(means, weights):
    # Note that the input weights are logits
    weighted_means = torch.einsum("bnkd,bnk->bnd", means, weights.softmax(-1))
    return weighted_means


def gmm_sample_weighted(means, variances, weights):
    # Note that the input weights are logits
    # gmm_check_params(means, variances, weights)

    # Reshape means and variances
    std = torch.sqrt(variances)
    normal_dist = D.Normal(means, std)
    samples = normal_dist.sample()  # Shape: (batch_size, num_tokens, k, d)

    probs = weights.softmax(-1)
    samples = (samples * probs.unsqueeze(-1)).sum(-2)

    return samples


def gmm_sample(means, variances, weights):
    # Note that the input weights are logits
    # gmm_check_params(means, variances, weights)

    batch_size, num_tokens, k, d = means.shape

    # Reshape weights and sample component indices
    weights_flat = weights.view(-1, k)  # Flatten to (batch_size * num_tokens, k)
    mixture_dist = D.Categorical(logits=weights_flat)
    component_indices = mixture_dist.sample()  # Shape: (batch_size * num_tokens,)

    # Reshape means and variances and select based on sampled indices
    means_flat = means.view(-1, k, d)  # Flatten to (batch_size * num_tokens, k, d)
    variances_flat = variances.view(-1, k, d)  # Same as means

    # Use advanced indexing to select the means and variances
    means_selected = means_flat[
        torch.arange(means_flat.size(0)), component_indices
    ]  # Shape: (batch_size * num_tokens, d)
    variances_selected = variances_flat[
        torch.arange(variances_flat.size(0)), component_indices
    ]

    # Compute standard deviations and sample from the normal distributions
    std_selected = torch.sqrt(variances_selected)
    normal_dist = D.Normal(means_selected, std_selected)
    samples_flat = normal_dist.sample()  # Shape: (batch_size * num_tokens, d)

    # Reshape samples back to (batch_size, num_tokens, d)
    samples = samples_flat.view(batch_size, num_tokens, d)

    return samples


def gmm_params_to_predictions(params, k, do_sample=False, var_scale=1.0, weighted_sample=False):
    # The output of GMM loss is GMM params, we need to sample from it
    d = (params.shape[-1] - k) // k // 2
    assert 2 * k * d + k == params.shape[-1], \
        "Invalid GMM params, k = {}, 2 * k * d + k = {}".format(k, params.shape[-1])
    means, variances, weights = gmm_split_params(params, k, d, var_scale=var_scale)
    if do_sample:
        if weighted_sample:
            predictions = gmm_sample_weighted(means, variances, weights)
        else:
            predictions = gmm_sample(means, variances, weights)
    else:
        predictions = gmm_predict(means, weights)
    
    return predictions


class ContinuousLVLM_Video(nn.Module):
    def __init__(self, llm, input_resampler, output_resampler, 
    num_frames=4, query_pos=False, learnable_pos=False, use_kl=False,
    num_gmm_kernel=16, transformer_decode=False, lm_loss_scale=1.0,
    rec_loss_scale=0.0, l1_loss_scale=0.0, gmm_loss_scale=0.0,
    kde_loss_scale=0.0, flow_loss_scale=0.0, cos_loss_scale=0.0) -> None:
        super().__init__()
        self.llm = llm
        self.input_resampler = input_resampler
        self.output_resampler = output_resampler
        self.num_frames = num_frames
        self.query_pos = query_pos
        self.learnable_pos = learnable_pos
        self.num_frames = num_frames
        self.query_pos = query_pos
        self.learnable_pos = learnable_pos
        self.use_kl = use_kl
        self.lm_loss_scale = lm_loss_scale
        self.rec_loss_scale = rec_loss_scale
        self.cos_loss_scale = cos_loss_scale
        self.l1_loss_scale = l1_loss_scale
        self.gmm_loss_scale = gmm_loss_scale
        self.kde_loss_scale = kde_loss_scale
        self.flow_loss_scale = flow_loss_scale
        
        self.mse = torch.nn.MSELoss() 
        self.l1 = torch.nn.L1Loss()
        self.cos = False

        self.num_gmm_kernel = num_gmm_kernel
        self.transformer_decode = transformer_decode

        if self.query_pos:
            embed_dim = self.output_resampler.input_dim
            if not self.learnable_pos:
                grid_size = int(math.sqrt(int(self.output_resampler.num_queries / self.num_frames)))
                self.pos_embed_spatial = nn.Parameter(
                    torch.from_numpy(
                        get_2d_sincos_pos_embed(embed_dim, grid_size)
                    ).float()
                ).requires_grad_(False)

                self.pos_embed_temporal = nn.Parameter(
                    torch.from_numpy(
                        get_1d_sincos_pos_embed_from_grid(
                            embed_dim, np.arange(self.num_frames)
                        )
                    ).float()
                ).requires_grad_(False)
            else:
                print("learnable query positional embeddings")
                scale = embed_dim ** -0.5
                self.pos_embed_temporal = nn.Parameter(
                    scale * torch.randn(self.num_frames, embed_dim)
                ).requires_grad_(True)

                self.pos_embed_spatial = nn.Parameter(
                    scale * torch.randn(
                        int(self.output_resampler.num_queries / self.num_frames), 
                        embed_dim
                    )
                ).requires_grad_(True)
            
        if self.transformer_decode:
            self.transformer_depth = 4
            self.dim = embed_dim
            self.pos_embed_decoder = nn.Parameter(scale * torch.randn(self.output_resampler.num_queries, self.dim))
            self.decoder_blocks = nn.ModuleList([
                Block(
                    dim=self.dim, 
                    num_heads=8, 
                    mlp_ratio=4.0, 
                    qkv_bias=True, 
                    qk_scale=None,
                    drop=0.0, 
                    attn_drop=0.0, 
                    drop_path=0.0, 
                    norm_layer=partial(nn.LayerNorm, eps=1e-6), 
                    use_grad_checkpointing=False
                )
                for i in range(self.transformer_depth)
            ])


    def get_video_embeds_gt(self, video_embeds, gen_turn_id):
        # Initialize a tensor to store the ground truth video embeddings.
        # The shape is the same as the input video_embeds but only keeps the temporal dimension.
        video_embeds_gt = torch.zeros(
            video_embeds.shape[0], 
            video_embeds.shape[2], 
            video_embeds.shape[3]
        ).to(video_embeds.device).to(video_embeds.dtype)    
            
        # Loop over each example in the batch.
        for i in range(gen_turn_id.shape[0]):
            # Retrieve the specific turn index for the current video from gen_turn_id.
            j = gen_turn_id[i].item()
            # Select the embedding for the j-th turn and assign it to the ground truth tensor.
            video_embeds_gt[i] = video_embeds[i, j]
        
        # Return the tensor containing the selected ground truth video embeddings.
        return video_embeds_gt

    def update_input_embeds(self, input_embeds, video_embeds_lm, ids_cmp_mask):
        # Loop over each example in the batch.
        for i in range(input_embeds.shape[0]):
            # Get the comparison mask for the current input embeddings.
            mask = ids_cmp_mask[i]
            # Find the indices where the mask is True.
            true_indices = torch.where(mask)[0]
            
            # If there are no true indices, skip updating and continue to the next iteration.
            if len(true_indices) == 0:
                input_embeds[i, :video_embeds_lm.shape[2], :] = (
                    input_embeds[i, :video_embeds_lm.shape[2], :] + 
                    0.0 * video_embeds_lm[i, 0, :]
                )
                continue
            
            # Initialize a list to hold groups of consecutive true indices.
            groups = []
            current_group = [true_indices[0]]

            # Group consecutive indices together.
            for idx in range(1, len(true_indices)):
                if true_indices[idx] == true_indices[idx - 1] + 1:
                    current_group.append(true_indices[idx])
                else:
                    groups.append(current_group)
                    current_group = [true_indices[idx]]
            if current_group:
                groups.append(current_group)

            # Loop over each group and update the corresponding segment of input embeddings.
            for j, group in enumerate(groups):
                if j < video_embeds_lm.shape[1]:  # Ensure we do not exceed the number of available video embeddings.
                    start_idx = group[0]
                    end_idx = group[-1] + 1
                    # Update the input embeddings with the corresponding video embeddings.
                    input_embeds[i, start_idx:end_idx] = video_embeds_lm[i, j]
        
        # Return the updated input embeddings.
        return input_embeds

    def update_input_embeds_face(self, input_embeds,input_ids,video_embeds_lm):
        for i in range(input_embeds.shape[0]):
            mask = (input_ids[i] == 32286)
            find_face = torch.nonzero(mask).squeeze()
            indices = find_face[0].item()
            input_embeds[i][indices:indices+len(find_face)] = video_embeds_lm[i]
        return input_embeds

    def forward(self, input_ids, attention_mask, labels, video_embeds, embeds_gen_mask, embeds_cmp_mask, ids_gen_mask,
                ids_cmp_mask,gen_turn_id,face_emb):
        
        input_embeds = self.llm.get_input_embeddings()(input_ids)  # bz x seq_len x dim

        bz, sq, dim = input_embeds.shape

        if self.query_pos:

            query_embeds = input_embeds[ids_gen_mask].reshape(
                input_embeds.shape[0], 
                self.num_frames, 
                -1, 
                input_embeds.shape[-1]
            )

            query_embeds_pos = (
                query_embeds + 
                self.pos_embed_spatial.unsqueeze(0).unsqueeze(1) + 
                self.pos_embed_temporal.unsqueeze(0).unsqueeze(2)
            )

            input_embeds[ids_gen_mask] = query_embeds_pos.reshape(
                -1, 
                query_embeds_pos.shape[-1]
            )
   
        if face_emb is not None:
            video_embeds_lm = self.input_resampler(face_emb.to(input_embeds.dtype))
            video_embeds_gt = self.get_video_embeds_gt(video_embeds,gen_turn_id)
            input_embeds = self.update_input_embeds_face(input_embeds,input_ids,video_embeds_lm)
        else:
            video_embeds_lm = self.input_resampler(video_embeds)
            video_embeds_gt = self.get_video_embeds_gt(video_embeds,gen_turn_id)
            input_embeds = self.update_input_embeds(input_embeds,video_embeds_lm,ids_cmp_mask)
        output_lm = self.llm(attention_mask=attention_mask,
                             inputs_embeds=input_embeds,
                             labels=labels,
                             output_hidden_states=True,
                             return_dict=True)
        lm_loss = output_lm['loss']
        # Training Forward Here
        last_hidden_state = output_lm.hidden_states[-1] 
        target_embeds = video_embeds_gt[embeds_gen_mask]  # num_imgs_gen_target x nq_in x dim_in
        num_imgs_for_rec = target_embeds.shape[0]
        output_image_embeds = last_hidden_state[ids_gen_mask].view(num_imgs_for_rec, -1, dim) 
        
        if self.transformer_decode:
            output_image_embeds = output_image_embeds + self.pos_embed_decoder.unsqueeze(0)
            for decoder_blk in self.decoder_blocks:
                output_image_embeds = decoder_blk(output_image_embeds)
        recon_image_embeds = self.output_resampler(output_image_embeds) 
        
        if self.rec_loss_scale != 0.0:
            rec_loss = self.mse(recon_image_embeds, target_embeds)
            
            if self.cos_loss_scale != 0.0:
                cos_loss = cosine_loss(recon_image_embeds, target_embeds)
                total_loss = (
                    self.lm_loss_scale * lm_loss + 
                    self.rec_loss_scale * rec_loss + 
                    self.cos_loss_scale * cos_loss
                )
                return {
                    'total_loss': total_loss, 
                    'lm_loss': lm_loss, 
                    'rec_loss': rec_loss, 
                    'cos_loss': cos_loss
                }
            else:
                total_loss = self.lm_loss_scale * lm_loss + self.rec_loss_scale * rec_loss
                return {
                    'total_loss': total_loss, 
                    'lm_loss': lm_loss, 
                    'rec_loss': rec_loss
                }

        else:
            total_loss = self.lm_loss_scale * lm_loss
            return {
                'total_loss': total_loss, 
                'lm_loss': lm_loss
            }
            
    def generate(self,
                 tokenizer,
                 prompt=None,
                 input_ids=None,
                 video_embeds=None,
                 embeds_cmp_mask=None,
                 ids_cmp_mask=None,
                 gen_turn_id=None,
                 face_tensor=None,
                 logits_processor=None,
                 num_vid_gen_tokens=100,
                 temperature=0.7,
                 num_beams=1,
                 max_new_tokens=520,
                 top_p=0.5,
                 device="cuda"):
        if logits_processor is None:
            logits_processor = LogitsProcessorList()
            logits_processor.append(
                AutoVideoTokenGenerationProcessor(tokenizer=tokenizer, num_vid_gen_tokens=num_vid_gen_tokens))

        if prompt is not None:
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids

        if isinstance(input_ids, list):
            input_ids = torch.tensor(input_ids)

        input_ids = input_ids.to(device=device)
        input_embeds = self.llm.get_input_embeddings()(input_ids)
        bz, sq, dim = input_embeds.shape
        
        if video_embeds is not None:
            with torch.no_grad():
                if face_tensor is not None:
                    video_embeds_lm = self.input_resampler(face_tensor.to(input_embeds.dtype))
                    video_embeds_gt = self.get_video_embeds_gt(video_embeds,gen_turn_id)
                    input_embeds = self.update_input_embeds_face(input_embeds,input_ids,video_embeds_lm)
                else:
                    video_embeds_lm = self.input_resampler(video_embeds)
                    video_embeds_gt = self.get_video_embeds_gt(video_embeds,gen_turn_id)
                    input_embeds = self.update_input_embeds(input_embeds,video_embeds_lm,ids_cmp_mask)
                
        generation_config = {
            'temperature': temperature,
            'num_beams': num_beams,
            'max_new_tokens': max_new_tokens,
            'top_p': top_p,
            'do_sample': False
        }

        # generate_ids = self.llm.generate(input_ids=input_ids, **generation_config)
        output = self.llm.generate(input_ids=input_ids,
                                   inputs_embeds=input_embeds,
                                   output_hidden_states=True,
                                   return_dict_in_generate=True,
                                   logits_processor=logits_processor,
                                   **generation_config)
                                   
        generate_ids = output.sequences[0][input_ids.shape[1]:]
        boi_token_id = tokenizer.encode(BOI_TOKEN, add_special_tokens=False)[0]
        eoi_token_id = tokenizer.encode(EOI_TOKEN, add_special_tokens=False)[0]

        last_hidden_states = torch.cat([hidden_state[-1] for hidden_state in output.hidden_states],
                                       dim=1)[0, input_ids.shape[1]:, :]

        eoi_indices = torch.where(generate_ids == eoi_token_id)[0].tolist()
        num_gen_vids = len(eoi_indices)
        text_mask = torch.ones_like(generate_ids, dtype=torch.bool)
        has_vid_output = num_gen_vids > 0
        if num_gen_vids > 0:
            vid_gen_feats = []
            for eoi_idx in eoi_indices:
                vid_gen_feats.append(last_hidden_states[eoi_idx - num_vid_gen_tokens:eoi_idx])
                text_mask[eoi_idx - num_vid_gen_tokens:eoi_idx] = False

            vid_gen_feats = torch.stack(vid_gen_feats)
            vid_gen_feat = self.output_resampler(vid_gen_feats)
        else:
            vid_gen_feat = None

        text_mask[generate_ids == boi_token_id] = False
        generate_ids = generate_ids[text_mask]
        generate_text = tokenizer.decode(generate_ids, skip_special_tokens=False)

        return {
            'text': generate_text,
            'has_vid_output': has_vid_output,
            'vid_gen_feat': vid_gen_feat,
            'num_gen_vids': num_gen_vids
        }

    def generate_bi(self,
                 tokenizer,
                 prompt=None,
                 input_ids=None,
                 attention_mask=None,
                 ids_gen_mask=None,
                 video_embeds=None,
                 embeds_cmp_mask=None,
                 ids_cmp_mask=None,
                 gen_turn_id=None,
                 face_tensor=None,
                 logits_processor=None,
                 num_vid_in_tokens=100,
                 num_vid_out_tokens=100,
                 var_scale=0.1,
                 temperature=0.7,
                 num_beams=1,
                 max_new_tokens=520,
                 top_p=0.5,
                 device="cuda"):
        if logits_processor is None:
            logits_processor = LogitsProcessorList()
            logits_processor.append(
                AutoVideoTokenGenerationProcessor(tokenizer=tokenizer, num_vid_gen_tokens=num_vid_in_tokens))

        if prompt is not None:
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids

        if isinstance(input_ids, list):
            input_ids = torch.tensor(input_ids)

        input_ids = input_ids.to(device=device)
        attention_mask = attention_mask.to(device=device)
        input_embeds = self.llm.get_input_embeddings()(input_ids) # [1, 74, 4096]
        bz, sq, dim = input_embeds.shape

        if video_embeds is not None:
            with torch.no_grad():
                if face_tensor is not None:
                    video_embeds_lm = self.input_resampler(face_tensor.to(input_embeds.dtype))
                    video_embeds_gt = self.get_video_embeds_gt(video_embeds,gen_turn_id)
                    input_embeds = self.update_input_embeds_face(input_embeds,input_ids,video_embeds_lm)
                else:
                    video_embeds_lm = self.input_resampler(video_embeds)
                    video_embeds_gt = self.get_video_embeds_gt(video_embeds,gen_turn_id)
                    input_embeds = self.update_input_embeds(input_embeds,video_embeds_lm,ids_cmp_mask)
                            

        if ids_gen_mask is not None and self.query_pos:
            query_embeds = input_embeds[ids_gen_mask].reshape(
                input_embeds.shape[0], 
                self.num_frames, 
                -1, 
                input_embeds.shape[-1]
            )

            query_embeds_pos = (
                query_embeds + 
                self.pos_embed_spatial.unsqueeze(0).unsqueeze(1) + 
                self.pos_embed_temporal.unsqueeze(0).unsqueeze(2)
            )

            input_embeds[ids_gen_mask] = query_embeds_pos.reshape(
                -1, 
                query_embeds_pos.shape[-1]
            )

        output = self.llm(
            inputs_embeds=input_embeds,
            return_dict=True,
            use_cache=True,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        output_embeds = output.hidden_states[-1][:,-num_vid_in_tokens:]
        
        print(input_embeds.shape, output_embeds.shape)

        if self.transformer_decode:
            output_embeds = output_embeds + self.pos_embed_decoder.unsqueeze(0)
            for decoder_blk in self.decoder_blocks:
                output_embeds = decoder_blk(output_embeds)
                
        vid_gen_feats = self.output_resampler(output_embeds)

        return vid_gen_feats

    @classmethod
    def from_pretrained(cls, llm, input_resampler, output_resampler, pretrained_model_path=None, **kwargs):
        model = cls(llm=llm, input_resampler=input_resampler, output_resampler=output_resampler, **kwargs)
        if pretrained_model_path is not None:
            ckpt = torch.load(pretrained_model_path, map_location='cpu')
            missing, unexpected = model.load_state_dict(ckpt, strict=False)
            print('agent model, missing keys: ', len(missing), 'unexpected keys:', len(unexpected))
            print('agent model, missing keys: ', missing)
            print('unexpected keys:', unexpected)
        return model
    