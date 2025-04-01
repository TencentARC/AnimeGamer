import os
import math
import argparse
from typing import List, Union
from tqdm import tqdm
from omegaconf import ListConfig
import imageio
import torch
import numpy as np
from einops import rearrange
import torchvision.transforms as TT
from sat.model.base_model import get_model
from sat.training.model_io import load_checkpoint
from sat import mpu
from VDM_Decoder.diffusion_video import SATVideoDiffusionEngine
from VDM_Decoder.arguments import get_args
from torchvision.transforms.functional import center_crop, resize
from torchvision.transforms import InterpolationMode
from PIL import Image
import random
from decord import VideoReader
import decord
import torchvision.transforms as transforms
import numpy as np


def cuculate_level(tire):
    motion = int(tire)
    if motion == 1:
        tire = 0.5
    elif motion == 2:
        tire = 2.5
    elif motion == 3:
        tire = 10
    elif motion == 4:
        tire = 15
    else:
        tire = 40
    return tire

def read_labels_and_videos(folder_path):
    labels_folder = os.path.join(folder_path, 'labels')
    videos_folder = os.path.join(folder_path, 'videos')
    results = []
    if not os.path.exists(labels_folder) or not os.path.exists(videos_folder):
        raise FileNotFoundError("Either 'labels' or 'videos' folder does not exist.")

    label_files = sorted(os.listdir(labels_folder), key=lambda x: int(os.path.splitext(x)[0]))

    transform = transforms.Compose([
        transforms.ToTensor()  
    ])

    
    for label_file in label_files:
        if label_file.endswith('.txt'):
            label_path = os.path.join(labels_folder, label_file)
            video_file = label_file.replace('.txt', '.mp4')
            video_path = os.path.join(videos_folder, video_file)
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video file {video_file} does not exist.")
            with open(label_path, 'r', encoding='utf-8') as f:
                label_content = f.read()
            video_abs_path = os.path.abspath(video_path)

            results.append((label_content, video_abs_path))

    return results

def read_from_cli():
    cnt = 0
    try:
        while True:
            x = input("Please input English text (Ctrl-D quit): ")
            yield x.strip(), cnt
            cnt += 1
    except EOFError as e:
        pass


def read_from_file(p, rank=0, world_size=1):
    with open(p, "r") as fin:
        cnt = -1
        for l in fin:
            cnt += 1
            if cnt % world_size != rank:
                continue
            yield l.strip(), cnt


def get_unique_embedder_keys_from_conditioner(conditioner):
    return list(set([x.input_key for x in conditioner.embedders]))


def get_batch(keys, value_dict, N: Union[List, ListConfig], T=None, device="cuda"):
    batch = {}
    batch_uc = {}

    for key in keys:
        if key == "txt":
            batch["txt"] = np.repeat([value_dict["prompt"]], repeats=math.prod(N)).reshape(N).tolist()
            batch_uc["txt"] = np.repeat([value_dict["negative_prompt"]], repeats=math.prod(N)).reshape(N).tolist()
        else:
            batch[key] = value_dict[key]

    if T is not None:
        batch["num_video_frames"] = T

    for key in batch.keys():
        if key not in batch_uc and isinstance(batch[key], torch.Tensor):
            batch_uc[key] = torch.clone(batch[key])
    return batch, batch_uc


def save_video_as_grid_and_mp4(video_batch: torch.Tensor, save_path: str, fps: int = 5, args=None, key=None):
    # os.makedirs(save_path, exist_ok=True)

    for i, vid in enumerate(video_batch):
        gif_frames = []
        for frame in vid:
            frame = rearrange(frame, "c h w -> h w c")
            frame = (255.0 * frame).cpu().numpy().astype(np.uint8)
            gif_frames.append(frame)
        now_save_path = f"{save_path}.mp4"
        with imageio.get_writer(now_save_path, fps=fps) as writer:
            for frame in gif_frames:
                writer.append_data(frame)


def resize_for_rectangle_crop(arr, image_size, reshape_mode="random"):
    if arr.shape[3] / arr.shape[2] > image_size[1] / image_size[0]:
        arr = resize(
            arr,
            size=[image_size[0], int(arr.shape[3] * image_size[0] / arr.shape[2])],
            interpolation=InterpolationMode.BICUBIC,
        )
    else:
        arr = resize(
            arr,
            size=[int(arr.shape[2] * image_size[1] / arr.shape[3]), image_size[1]],
            interpolation=InterpolationMode.BICUBIC,
        )

    h, w = arr.shape[2], arr.shape[3]
    arr = arr.squeeze(0)

    delta_h = h - image_size[0]
    delta_w = w - image_size[1]

    if reshape_mode == "random" or reshape_mode == "none":
        top = np.random.randint(0, delta_h + 1)
        left = np.random.randint(0, delta_w + 1)
    elif reshape_mode == "center":
        top, left = delta_h // 2, delta_w // 2
    else:
        raise NotImplementedError
    arr = TT.functional.crop(arr, top=top, left=left, height=image_size[0], width=image_size[1])
    return arr


def sampling_main(args, model_cls):
    decord.bridge.set_bridge("torch")
    if isinstance(model_cls, type):
        model = get_model(args, model_cls)
    else:
        model = model_cls

    load_checkpoint(model, args)
    model.eval()

    if args.input_type == "cli":
        data_iter = read_from_cli()
    elif args.input_type == "txt":
        rank, world_size = mpu.get_data_parallel_rank(), mpu.get_data_parallel_world_size()
        print("rank and world_size", rank, world_size)
        data_iter = read_from_file(args.input_file, rank=rank, world_size=world_size)
    else:
        raise NotImplementedError

    image_size = [480, 720]

    if args.image2video:
        chained_trainsforms = []
        chained_trainsforms.append(TT.ToTensor())
        transform = TT.Compose(chained_trainsforms)

    sample_func = model.sample
    T, H, W, C, F = args.sampling_num_frames, image_size[0], image_size[1], args.latent_channels, 8
    num_samples = [1]
    force_uc_zero_embeddings = ["txt"]
    device = model.device

    ARresult = torch.load(args.LLM_output_path)

    with torch.no_grad():
        for key, value in ARresult.items():
            count = key.split('_turn_')[0][3:]
            batch = {'txt': ['']}
            batch_uc = {'txt': ['']}

            c, uc = model.conditioner.get_unconditional_conditioning(
                batch, # {'txt': ['bala bala']}
                batch_uc=batch_uc, # {'txt': ['']}
                force_uc_zero_embeddings=force_uc_zero_embeddings,
            )

            for k in c:
                if not k == "crossattn":
                    c[k], uc[k] = map(lambda y: y[k][: math.prod(num_samples)].to("cuda"), (c, uc))

            c['ip_cond'] = torch.zeros((1, 196, 768)).type(torch.float16).cuda()
            uc['ip_cond'] = torch.zeros_like(c['ip_cond'])

            c['face_id_cond'] = torch.zeros_like(c['ip_cond'])
            uc['face_id_cond'] = torch.zeros_like(c['ip_cond'])
            flow_number = cuculate_level(int(value[2][0]))
            for index in [flow_number]:
                model.to(device)
                samples_z = sample_func(
                    c,
                    uc=uc,
                    batch_size=1,
                    shape=(T, C, H // F, W // F),
                    flow=torch.tensor(flow_number),
                    aroutput=value[0].type(torch.bfloat16).cuda(), #[1, 226, 1920]
                )
                samples_z = samples_z.permute(0, 2, 1, 3, 4).contiguous()

                # Unload the model from GPU to save GPU memory
                # model.to("cpu")
                torch.cuda.empty_cache()
                first_stage_model = model.first_stage_model
                first_stage_model = first_stage_model.to(device)

                latent = 1.0 / model.scale_factor * samples_z

                # Decode latent serial to save GPU memory
                os.makedirs(args.output_dir, exist_ok=True)
                recons = []
                loop_num = (T - 1) // 2
                for i in range(loop_num):
                    if i == 0:
                        start_frame, end_frame = 0, 3
                    else:
                        start_frame, end_frame = i * 2 + 1, i * 2 + 3
                    if i == loop_num - 1:
                        clear_fake_cp_cache = True
                    else:
                        clear_fake_cp_cache = False
                    with torch.no_grad():
                        recon = first_stage_model.decode(
                            latent[:, :, start_frame:end_frame].contiguous(), clear_fake_cp_cache=clear_fake_cp_cache
                        )

                    recons.append(recon)

                recon = torch.cat(recons, dim=2).to(torch.float32)
                samples_x = recon.permute(0, 2, 1, 3, 4).contiguous()
                samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0).cpu()

                save_path = os.path.join(
                    args.output_dir, str(count)
                )
                if mpu.get_model_parallel_rank() == 0:
                    save_video_as_grid_and_mp4(samples, save_path, fps=args.sampling_fps)


if __name__ == "__main__":
    if "OMPI_COMM_WORLD_LOCAL_RANK" in os.environ:
        os.environ["LOCAL_RANK"] = os.environ["OMPI_COMM_WORLD_LOCAL_RANK"]
        os.environ["WORLD_SIZE"] = os.environ["OMPI_COMM_WORLD_SIZE"]
        os.environ["RANK"] = os.environ["OMPI_COMM_WORLD_RANK"]
    py_parser = argparse.ArgumentParser(add_help=False)
    known, args_list = py_parser.parse_known_args()

    args = get_args(args_list)
    args = argparse.Namespace(**vars(args), **vars(known))
    del args.deepspeed_config
    args.model_config.first_stage_config.params.cp_size = 1
    args.model_config.network_config.params.transformer_args.model_parallel_size = 1
    args.model_config.network_config.params.transformer_args.checkpoint_activations = False
    args.model_config.loss_fn_config.params.sigma_sampler_config.params.uniform_sampling = False

    sampling_main(args, model_cls=SATVideoDiffusionEngine)
