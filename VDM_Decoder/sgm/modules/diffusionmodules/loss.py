from typing import List, Optional, Union

import torch
import torch.nn as nn
from omegaconf import ListConfig
from ...util import append_dims, instantiate_from_config
from ...modules.autoencoding.lpips.loss.lpips import LPIPS
from sat import mpu
import random

class StandardDiffusionLoss(nn.Module):
    def __init__(
        self,
        sigma_sampler_config,
        type="l2",
        offset_noise_level=0.0,
        batch2model_keys: Optional[Union[str, List[str], ListConfig]] = None,
        do_training_cfg=False,
    ):
        super().__init__()

        assert type in ["l2", "l1", "lpips"]

        self.sigma_sampler = instantiate_from_config(sigma_sampler_config)

        self.type = type
        self.offset_noise_level = offset_noise_level

        if type == "lpips":
            self.lpips = LPIPS().eval()

        if not batch2model_keys:
            batch2model_keys = []

        if isinstance(batch2model_keys, str):
            batch2model_keys = [batch2model_keys]

        self.batch2model_keys = set(batch2model_keys)
        self.do_training_cfg = do_training_cfg

    def __call__(self, network, denoiser, conditioner, input, batch):
        cond = conditioner(batch)
        additional_model_inputs = {key: batch[key] for key in self.batch2model_keys.intersection(batch)}

        sigmas = self.sigma_sampler(input.shape[0]).to(input.device)
        noise = torch.randn_like(input)
        if self.offset_noise_level > 0.0:
            noise = (
                noise + append_dims(torch.randn(input.shape[0]).to(input.device), input.ndim) * self.offset_noise_level
            )
            noise = noise.to(input.dtype)
        noised_input = input.float() + noise * append_dims(sigmas, input.ndim)
        model_output = denoiser(network, noised_input, sigmas, cond, **additional_model_inputs)
        w = append_dims(denoiser.w(sigmas), input.ndim)
        return self.get_loss(model_output, input, w)

    def get_loss(self, model_output, target, w):
        if self.type == "l2":
            return torch.mean((w * (model_output - target) ** 2).reshape(target.shape[0], -1), 1)
        elif self.type == "l1":
            return torch.mean((w * (model_output - target).abs()).reshape(target.shape[0], -1), 1)
        elif self.type == "lpips":
            loss = self.lpips(model_output, target).reshape(-1)
            return loss


class VideoDiffusionLoss(StandardDiffusionLoss): # Main training code
    def __init__(self, block_scale=None, block_size=None, min_snr_value=None, fixed_frames=0, **kwargs):
        self.fixed_frames = fixed_frames
        self.block_scale = block_scale
        self.block_size = block_size
        self.min_snr_value = min_snr_value
        super().__init__(**kwargs)

    def __call__(self, network, denoiser, conditioner, clip_img_encoder, input, batch):

        if self.do_training_cfg:
            do_cfg_txt = random.random()
            if do_cfg_txt <= 0.05:
                batch['txt'] = ''

        cond = conditioner(batch)
        ip_cond = clip_img_encoder(batch['ip_img'])
        # cond['face_id_cond'] = False

        if 'face_id' in batch.keys():
            if len(batch['face_id']) != 0:
                for i in range(len(batch['face_id'])):
                    if i == 0 :
                        face_id_cond = clip_img_encoder(batch['face_id'][i].to(input.device))
                    else:
                        face_id_cond = torch.cat((face_id_cond, clip_img_encoder(batch['face_id'][i].to(input.device))), dim=1)
                cond['face_id_cond'] = face_id_cond
        
        cond['ip_cond'] = ip_cond

        if self.do_training_cfg:
            do_cfg_img = random.random()
            if do_cfg_img <= 0.05:
                cond['ip_cond'] = torch.zeros_like(cond['ip_cond'])


        additional_model_inputs = {key: batch[key] for key in self.batch2model_keys.intersection(batch)}

        alphas_cumprod_sqrt, idx = self.sigma_sampler(input.shape[0], return_idx=True)
        alphas_cumprod_sqrt = alphas_cumprod_sqrt.to(input.device)
        idx = idx.to(input.device)

        noise = torch.randn_like(input)

        # broadcast noise
        mp_size = mpu.get_model_parallel_world_size()
        global_rank = torch.distributed.get_rank() // mp_size
        src = global_rank * mp_size
        torch.distributed.broadcast(idx, src=src, group=mpu.get_model_parallel_group())
        torch.distributed.broadcast(noise, src=src, group=mpu.get_model_parallel_group())
        torch.distributed.broadcast(alphas_cumprod_sqrt, src=src, group=mpu.get_model_parallel_group())

        additional_model_inputs["idx"] = idx
        additional_model_inputs["flow_idx"] = batch['motion'] # flow condition
        if 'AR_mixemb' in batch.keys():
            additional_model_inputs["AR_mixemb"] = batch['AR_mixemb']

        if self.offset_noise_level > 0.0:
            noise = (
                noise + append_dims(torch.randn(input.shape[0]).to(input.device), input.ndim) * self.offset_noise_level
            )

        noised_input = input.float() * append_dims(alphas_cumprod_sqrt, input.ndim) + noise * append_dims(
            (1 - alphas_cumprod_sqrt**2) ** 0.5, input.ndim
        )

        if "concat_images" in batch.keys():
            cond["concat"] = batch["concat_images"]

        # [2, 13, 16, 60, 90],[2] dict_keys(['crossattn', 'concat'])  dict_keys(['idx'])
        #print(f"input time: {additional_model_inputs['idx']}")
        #print(f"input motion: {additional_model_inputs['flow_idx']}")

        model_output = denoiser(network, noised_input, alphas_cumprod_sqrt, cond, **additional_model_inputs)
        w = append_dims(1 / (1 - alphas_cumprod_sqrt**2), input.ndim)  # v-pred

        if self.min_snr_value is not None:
            w = min(w, self.min_snr_value)
        return self.get_loss(model_output, input, w)

    def get_loss(self, model_output, target, w):
        if self.type == "l2":
            return torch.mean((w * (model_output - target) ** 2).reshape(target.shape[0], -1), 1)
        elif self.type == "l1":
            return torch.mean((w * (model_output - target).abs()).reshape(target.shape[0], -1), 1)
        elif self.type == "lpips":
            loss = self.lpips(model_output, target).reshape(-1)
            return loss
