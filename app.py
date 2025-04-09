import gradio as gr
import numpy as np
import os

from pathlib import Path
import hydra
from omegaconf import OmegaConf
import torch
import re
import json
import pyrootutils
from PIL import Image
from tqdm import tqdm
import copy
import math
import argparse
from typing import List, Union
from omegaconf import ListConfig
import imageio
from einops import rearrange
import torchvision.transforms as TT
from sat.model.base_model import get_model
from sat.training.model_io import load_checkpoint
from sat import mpu
from VDM_Decoder.diffusion_video import SATVideoDiffusionEngine
from VDM_Decoder.arguments import get_args
from torchvision.transforms.functional import center_crop, resize
from torchvision.transforms import InterpolationMode
import random
from decord import VideoReader
import decord
import torchvision.transforms as transforms
DEFAULT_VIDEO_DIR = "./results/gradio_demo"
LOW_VRAM_VERSION = True

os.makedirs(DEFAULT_VIDEO_DIR, exist_ok=True)

# Define special tokens for different multimedia elements
MM_BOI_TOKEN = "<img>"
MM_EOI_TOKEN = "</img>"
BOI_TOKEN = "<frame>"
EOI_TOKEN = "</frame>"
FRAME_TOKEN = '<frame_{:05d}>'
STTOKEN = '<Strength_{:d}>'
WORLD_BML_TOKEN = '<ML>'
WORLD_EML_TOKEN = '</ML>'
WORLD_BSC_TOKEN = '<SC>'
WORLD_ESC_TOKEN = '</SC>'
WORLD_BET_TOKEN = '<ET>'
WORLD_EET_TOKEN = '</ET>'
WORLD_BST_TOKEN = '<ST>'
WORLD_EST_TOKEN = '</ST>'
WORLD_BTURN_TOKEN = '<TURN>'
WORLD_ETURN_TOKEN = '</TURN>'

# Regular expressions for parsing
op2 = r'(.*?)</ML>'
op3 = r'<SC>(.*?)</SC>'
op4 = r'<ET>(.*?)</ET>'
op5 = r'<ST>(.*?)</ST>'
op6 = r'<ML>(.*?)</ML>'



def cuculate_level(tire):
    motion = int(tire)
    if motion == 1:
        tire = 0.5
    elif motion == 2:
        tire = 1
    elif motion == 3:
        tire = 5
    elif motion == 4:
        tire = 20
    else:
        tire = 40
    return tire

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

def save_video_as_grid_and_mp4(video_batch: torch.Tensor, save_path: str, fps: int = 5, args=None, key=None):
    # os.makedirs(save_path, exist_ok=True)

    for i, vid in enumerate(video_batch):
        gif_frames = []
        for frame in vid:
            frame = rearrange(frame, "c h w -> h w c")
            frame = (255.0 * frame).cpu().numpy().astype(np.uint8)
            gif_frames.append(frame)

        with imageio.get_writer(save_path, fps=fps) as writer:
            for frame in gif_frames:
                writer.append_data(frame)


class AnimeGamer:
    def __init__(self) -> None:
        self.dtype = torch.bfloat16
        self.dtype_str = 'bf16'
        global LOW_VRAM_VERSION
        if LOW_VRAM_VERSION:
            self.device_vdm = 'cuda:0'
            self.device_mllm = 'cuda:1'
        else:
            self.device_vdm = 'cuda:0'
            self.device_mllm = 'cuda:0'            
        self.num_mm_tokens = 226
        self.tokenizer_cfg_path = './MLLM/configs/tokenizer/tokenizer.yaml'
        self.llm_cfg_path = './MLLM/configs/clm_models/mistral-7b.yaml'
        self.agent_cfg_path = './MLLM/configs/clm_models/animegamer_clm.yaml'
        self.max_window = 1
        self.step_by_step = True
        
        # Load MLLM
        tokenizer_cfg = OmegaConf.load(self.tokenizer_cfg_path)
        self.tokenizer = hydra.utils.instantiate(tokenizer_cfg)
        llm_cfg = OmegaConf.load(self.llm_cfg_path)
        self.llm = hydra.utils.instantiate(llm_cfg, torch_dtype=self.dtype)
        print('Init llm done.')
        agent_model_cfg = OmegaConf.load(self.agent_cfg_path)
        self.agent_model = hydra.utils.instantiate(agent_model_cfg, llm=self.llm)
        self.agent_model.eval().to(self.device_mllm, dtype=self.dtype)
        print('Init agent model Done')
        
        # Load VDM_Decoder
        if "OMPI_COMM_WORLD_LOCAL_RANK" in os.environ:
            os.environ["LOCAL_RANK"] = os.environ["OMPI_COMM_WORLD_LOCAL_RANK"]
            os.environ["WORLD_SIZE"] = os.environ["OMPI_COMM_WORLD_SIZE"]
            os.environ["RANK"] = os.environ["OMPI_COMM_WORLD_RANK"]
        py_parser = argparse.ArgumentParser(add_help=False)
        known, args_list = py_parser.parse_known_args()
        args = get_args(args_list)
        args = argparse.Namespace(**vars(args), **vars(known))
        del args.deepspeed_config
        args.device = int(self.device_vdm[-1])
        args.model_config.first_stage_config.params.cp_size = 1
        args.model_config.network_config.params.transformer_args.model_parallel_size = 1
        args.model_config.network_config.params.transformer_args.checkpoint_activations = False
        args.model_config.loss_fn_config.params.sigma_sampler_config.params.uniform_sampling = False
        model_cls = SATVideoDiffusionEngine
        decord.bridge.set_bridge("torch")
        if isinstance(model_cls, type):
            self.Decoder_model = get_model(args, model_cls)
        else:
            self.Decoder_model = model_cls
        load_checkpoint(self.Decoder_model, args)
        self.Decoder_model.eval()
        self.Decoder_model.to(self.device_vdm)
        self.image_size = [480, 720]
        self.sample_func = self.Decoder_model.sample
        self.T, self.H, self.W, self.C, self.F = args.sampling_num_frames, self.image_size[0], self.image_size[1], args.latent_channels, 8
        self.num_samples = [1]
        self.force_uc_zero_embeddings = ["txt"]
        self.sampling_fps = args.sampling_fps

animegamer = AnimeGamer()
global_stamina = 50
global_entertainment = 50
global_social = 50

def generate_MLLM(history, characters, motion_adverb, motion, time, background):    
    max_window = animegamer.max_window
    gen_turn_id = len(history)
    total_turn = 50
    if gen_turn_id == 0:
        character_states = [[0, 0, 0, 0] for i in range(total_turn)]
        mix_emb = torch.zeros([1, total_turn, 226, 1920]).to(animegamer.device_mllm).to(animegamer.dtype)    
        instructions = [f'Character: {characters}; Motion: {motion_adverb} {motion}; Background: {time} {background}.']
    else:
        instructions = history[-1]['instructions']
        instructions.append(f'Character: {characters}; Motion: {motion}; Background: {background}.')
        mix_emb = history[-1]['mix_emb']
        character_states = history[-1]['character_states']
        for item in history[:-1]: 
            if 'mix_emb' in item:
                del item['mix_emb']
            if 'character_states' in item:
                del item['character_states']

    input_ids = [animegamer.tokenizer.bos_token_id]
    ids_gen_mask = [False]
    ids_cmp_mask = [False]
    attention_mask = [1]
    emb_start = 0
    emb_end = emb_start + 1

    if gen_turn_id > 0:
        for turn_locate in range(gen_turn_id):

            if turn_locate >= max_window-1:
                break
            if gen_turn_id >= max_window:
                emb_start = gen_turn_id - max_window + 1
                emb_end = emb_start + max_window
                turn = turn_locate + emb_start
            else:
                turn = turn_locate
                emb_start = 0
                emb_end = emb_start + max_window  

            def format_world_strength(index):
                return STTOKEN.format(character_states[gen_turn_id][index])

            caption = instructions[turn]
            SC = character_states[turn][0]
            ET = character_states[turn][1]
            ST = character_states[turn][2]
            ML = character_states[turn][3]
            caption = WORLD_BTURN_TOKEN + caption
            caption_ids = animegamer.tokenizer.encode(caption, add_special_tokens=False)

            # Construct the world tokens string
            world_tokens = (
                WORLD_BML_TOKEN +
                format_world_strength(0) +
                WORLD_EML_TOKEN + WORLD_BSC_TOKEN +
                format_world_strength(1) +
                WORLD_ESC_TOKEN + WORLD_BET_TOKEN +
                format_world_strength(2) +
                WORLD_EET_TOKEN + WORLD_BST_TOKEN +
                format_world_strength(3) +
                WORLD_EST_TOKEN + WORLD_EST_TOKEN +
                WORLD_ETURN_TOKEN
            )
        
            # Encode the world tokens
            world_ids = animegamer.tokenizer.encode(world_tokens, add_special_tokens=False)
            # Construct the video tokens string
            frame_tokens = ''.join([FRAME_TOKEN.format(int(item)) for item in range(animegamer.num_mm_tokens)])
            video_tokens = MM_BOI_TOKEN + (BOI_TOKEN + frame_tokens + EOI_TOKEN) * 1 + MM_EOI_TOKEN
            # Encode the video tokens
            video_ids = animegamer.tokenizer.encode(video_tokens, add_special_tokens=False)
            # Combine input IDs
            input_ids += caption_ids + video_ids + world_ids
            # Update attention mask
            attention_mask += [1] * len(caption_ids) + [1] * len(video_ids) + [1] * len(world_ids)
            # Update generation mask
            ids_gen_mask += [False] * len(caption_ids) + [False] * len(video_ids) + [False] * len(world_ids)
            # Update comparison mask
            video_cmp_mask = [False] + ([False] + [True] * animegamer.num_mm_tokens + [False]) * 1 + [False]
            ids_cmp_mask += [False] * len(caption_ids) + video_cmp_mask + [False] * len(world_ids)

    caption = instructions[gen_turn_id]
    caption = WORLD_BTURN_TOKEN + caption
    caption_ids = animegamer.tokenizer.encode(caption, add_special_tokens=False)
    world_tokens = WORLD_BML_TOKEN
    world_ids = animegamer.tokenizer.encode(world_tokens, add_special_tokens=False)            
    frame_sequence = ''.join([FRAME_TOKEN.format(int(item)) for item in range(animegamer.num_mm_tokens)])
    video_tokens = (
        MM_BOI_TOKEN +
        (BOI_TOKEN + frame_sequence + EOI_TOKEN) * 1 +
        MM_EOI_TOKEN
    )
    video_ids = animegamer.tokenizer.encode(video_tokens, add_special_tokens=False)
    input_ids += caption_ids + video_ids + world_ids
    caption_gen_mask = [False] * len(caption_ids)
    video_gen_mask = [False] + ([False] + [True] * animegamer.num_mm_tokens + [False]) * 1 + [False]
    world_gen_mask = [False] * len(world_ids)
    ids_gen_mask += caption_gen_mask + video_gen_mask + world_gen_mask
    caption_cmp_mask = [False] * len(caption_ids)
    video_cmp_mask = [False] * len(video_ids)
    world_cmp_mask = [False] * len(world_ids)
    ids_cmp_mask += caption_cmp_mask + video_cmp_mask + world_cmp_mask
    caption_attention = [1] * len(caption_ids)
    video_attention = [1] + ([1] + [-1] * animegamer.num_mm_tokens + [1]) * 1 + [1]
    world_attention = [1] * len(world_ids)
    attention_mask += caption_attention + video_attention + world_attention
    assert len(input_ids) == len(ids_gen_mask) == len(ids_cmp_mask)
    ids_gen_mask = torch.tensor(ids_gen_mask).to(animegamer.device_mllm, dtype=torch.bool).unsqueeze(0)
    attention_mask = torch.tensor(attention_mask).to(animegamer.device_mllm, dtype=torch.long).unsqueeze(0)
    input_ids = torch.tensor(input_ids).to(animegamer.device_mllm, dtype=torch.long).unsqueeze(0)
    ids_cmp_mask = torch.tensor(ids_cmp_mask).to(animegamer.device_mllm, dtype=torch.bool).unsqueeze(0)

    with torch.no_grad():

        mix_emb_now_input = mix_emb[:,emb_start:emb_end,:,:]
        if gen_turn_id < max_window:
            gen_id_tensor = torch.tensor(gen_turn_id).unsqueeze(0)
        else:
            gen_id_tensor = torch.tensor(max_window-1).unsqueeze(0)

        txt_output = animegamer.agent_model.generate(tokenizer=animegamer.tokenizer,
                                            input_ids=input_ids,
                                            video_embeds=mix_emb_now_input,
                                            gen_turn_id=gen_id_tensor,
                                            ids_cmp_mask=ids_cmp_mask,
                                            max_new_tokens=200,
                                            device=animegamer.device_mllm)['text']

        vid_output = animegamer.agent_model.generate_bi(tokenizer=animegamer.tokenizer, 
                                                input_ids=input_ids, 
                                                attention_mask=attention_mask, 
                                                ids_gen_mask=ids_gen_mask, 
                                                ids_cmp_mask=ids_cmp_mask,
                                                gen_turn_id=gen_id_tensor,
                                                num_vid_in_tokens=226, 
                                                video_embeds=mix_emb_now_input,
                                                num_vid_out_tokens=226, 
                                                device=animegamer.device_mllm)

    txt_out = txt_output.split(' ')
    try:
        pred_ML = int(txt_out[0][10:-1])
        pred_SC = int(txt_out[3][10:-1])
        pred_ET = int(txt_out[6][10:-1])
        pred_ST = int(txt_out[9][10:-1])
    except:
        pred_ML = 0
        pred_SC = 0
        pred_ET = 0
        pred_ST = 0
    pred_character_states = [pred_ML, pred_SC, pred_ET, pred_ST]
    if animegamer.step_by_step:
        mix_emb[0][gen_turn_id] = vid_output
        character_states[gen_turn_id] = pred_character_states
    torch.cuda.empty_cache()

    return vid_output, pred_character_states, mix_emb, character_states, instructions

def generate_Decoder(ml, vid_output, video_path):
        
    c = {"crossattn": torch.zeros((1, 30, 4096)).type(torch.float16).to(animegamer.device_vdm)}
    uc = {"crossattn": torch.zeros((1, 30, 4096)).type(torch.float16).to(animegamer.device_vdm)}
    c['ip_cond'] = torch.zeros((1, 196, 768)).type(torch.float16).to(animegamer.device_vdm)
    uc['ip_cond'] = torch.zeros_like(c['ip_cond'])
    c['face_id_cond'] = torch.zeros_like(c['ip_cond'])
    uc['face_id_cond'] = torch.zeros_like(c['ip_cond'])
    flow_number = cuculate_level(int(ml))

    for index in [flow_number]:
        samples_z = animegamer.sample_func(
            c,
            uc=uc,
            batch_size=1,
            shape=(animegamer.T, animegamer.C, animegamer.H // animegamer.F, animegamer.W // animegamer.F),
            flow=torch.tensor(flow_number),
            aroutput=vid_output, #[1, 226, 1920]
        )
        samples_z = samples_z.permute(0, 2, 1, 3, 4).contiguous()

        torch.cuda.empty_cache()
        first_stage_model = animegamer.Decoder_model.first_stage_model
        first_stage_model = first_stage_model.to(animegamer.device_vdm)

        latent = 1.0 / animegamer.Decoder_model.scale_factor * samples_z

        # Decode latent serial to save GPU memory
        recons = []
        loop_num = (animegamer.T - 1) // 2
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

        if mpu.get_model_parallel_rank() == 0:
            save_video_as_grid_and_mp4(samples, video_path, fps=animegamer.sampling_fps)


def generate_animation(history, characters, motion_adverb, motion, time, background, video_dir):
    Path(video_dir).mkdir(parents=True, exist_ok=True)
    video_filename = f"{len(history)+1}.mp4"
    video_path = os.path.abspath(os.path.join(video_dir, video_filename))

    vid_output, pred_character_states, mix_emb, character_states, instructions = generate_MLLM(history, characters, motion_adverb, motion, time, background)

    def constrain(value):
        return max(0, min(100, value))
    
    global global_social, global_stamina, global_entertainment
    
    global_social = constrain(global_social + (pred_character_states[1] * 5))
    global_stamina = constrain(global_stamina + (pred_character_states[3] * 5))
    global_entertainment = constrain(global_entertainment + (pred_character_states[2] * 5))
    
    stamina = copy.deepcopy(global_stamina)
    entertainment = copy.deepcopy(global_entertainment)
    social = copy.deepcopy(global_social)

    if pred_character_states[2] == 0:
        entertainment = entertainment - 5
    if pred_character_states[1] == 0:
        social = social - 5

    generate_Decoder(pred_character_states[1], vid_output.to(animegamer.device_vdm), video_path)

    print(social, stamina, entertainment)

    return mix_emb, video_path, stamina, entertainment, social, vid_output, character_states, instructions

def process_input(characters, motion_adverb, motion, time, background, video_dir, history):
    mix_emb, video_path, stamina, entertainment, social, vid_output, character_states, instructions = generate_animation(
        history=history,
        characters=characters,
        motion_adverb=motion_adverb,
        motion=motion,
        time=time,
        background=background,
        video_dir=video_dir,
    )
    
    new_entry = {
        "round": len(history) + 1,
        "characters": characters,
        "motion": f"{motion_adverb} {motion}",
        "background": f"{time} {background}",
        "video_path": video_path,
        "stamina": round(stamina, 1),
        "entertainment": round(entertainment, 1),
        "social": round(social, 1),
        "mix_emb": mix_emb,
        "character_states": character_states,
        "instructions": instructions
    }
    updated_history = history + [new_entry]
    
    history_html = generate_history_html(updated_history)
    return video_path, characters, motion_adverb, motion, time, background, updated_history, history_html

def get_color_class(value):
    if value < 30:
        return "low"
    elif 30 <= value <= 60:
        return "medium"
    else:
        return "high"

def generate_history_html(history):
    html = """
    <style>
        .video-cell {
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 10px 0 !important;
        }
        .video-cell video {
            display: block;
            margin: 0 auto;
        }

        .progress-bar-container {
            width: 150px;
            margin: 0 auto;
        }
        
        .progress-bar {
            width: 100%;  
            height: 20px;
            background: #eee;
            border-radius: 10px;
            overflow: hidden;
            position: relative;
        }

        .history-table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            border-radius: 8px;
            overflow: hidden;
        }
        .history-table th, .history-table td {
            padding: 12px 15px;
            text-align: center !important;
            vertical-align: middle !important;
            border-bottom: 1px solid #dddddd;
        }

        .history-table th {
            background-color: #f8f9fa;
            color: #2c3e50;
            font-weight: 600;
        }
        .progress-fill {
            height: 100%;
            transition: width 0.3s ease;
        }
        .value-text {
            position: absolute;
            width: 100%;
            text-align: center;
            font-size: 12px;
            color: #333;
            font-weight: bold;
        }
        .progress-fill.low { background: #ff4444; }
        .progress-fill.medium { background: #ffd700; }
        .progress-fill.high { background: #4CAF50; }
    </style>
    <table class="history-table">
        <tr>
            <th>Round</th>
            <th>Characters</th>
            <th>Motion</th>
            <th>Background</th>
            <th>Stamina</th>
            <th>Entertainment</th>
            <th>Social</th>
            <th>Video</th>
        </tr>
    """
    
    for entry in history:
        abs_video_path = os.path.abspath(entry['video_path'])
        video_url = f"/file={entry['video_path']}"  
        
        html += f"""
        <tr>
            <td>{entry['round']}</td>
            <td>{entry['characters']}</td>
            <td>{entry['motion']}</td>
            <td>{entry['background']}</td>
            {generate_progress_cell(entry['stamina'])}
            {generate_progress_cell(entry['entertainment'])}
            {generate_progress_cell(entry['social'])}
            <td class="video-cell">
                <video controls width="200" height="120">
                    <source src="{video_url}" type="video/mp4">
                </video>
            </td>
        </tr>
        """
    
    html += "</table>"

    return html

def generate_progress_cell(value):
    color_class = get_color_class(value)
    return f"""
    <td>
        <div class="progress-bar-container">
            <div class="progress-bar">
                <div class="progress-fill {color_class}" style="width: {value}%"></div>
                <div class="value-text">{value}%</div>
            </div>
        </div>
    </td>
    """

def restart_conversation(video_dir):
    Path(video_dir).mkdir(parents=True, exist_ok=True)

    return [
        [],                 # history
        None,               # video_output
        "Qiqi",             # characters
        "quickly",          # motion_adverb
        "fly on broomstick",# motion
        "day/night",        # time
        "sky",              # background
        "",                 # history_display
        video_dir,
    ]

with gr.Blocks(theme=gr.themes.Default()) as demo:
    state = gr.State()
    history = gr.State(value=[])
    
    title = r"""<h1 align="center">🪄💖 AnimeGamer: Infinite Anime Life Simulation with Next Game State Prediction 🧹🔮</h1>"""
    gr.Markdown(title)

    with gr.Row():
        with gr.Column(scale=5):
            video_dir = gr.Textbox(
                label="Video Output Directory",
                value=DEFAULT_VIDEO_DIR,
                placeholder="Enter output directory path..."
            )
            characters = gr.Textbox(label="Characters", value = "Qiqi")
            motion_adverb = gr.Textbox(label="Motion Adverb", value = "quickly")
            motion = gr.Textbox(label="Motion", value = "fly on broomstick")
            time = gr.Textbox(label="Time", value = "day/night")
            background = gr.Textbox(label="Background", value = "sky")
            
            with gr.Row():
                submit_btn = gr.Button("Generate Animation", variant="primary")
                restart_btn = gr.Button("Restart Conversation", variant="stop")
        
        with gr.Column(scale=5):
            video_output = gr.Video(
                label="Current Animation",
                format="mp4",
                interactive=False
            )
    
    with gr.Row():
        history_display = gr.HTML(
            label="Generation History",
            elem_classes="full-width"
        )

    submit_btn.click(
        fn=process_input,
        inputs=[characters, motion_adverb, motion, time, background, video_dir, history],
        outputs=[video_output, characters, motion_adverb, motion, time, background, history, history_display]
    )
    
    restart_btn.click(
        fn=restart_conversation,
        inputs=[video_dir],
        outputs=[history, video_output, characters, motion_adverb, motion, time, background, history_display, video_dir]
    )

css = """
.full-width {
    width: 100% !important;
}
"""
demo.css = css

if __name__ == "__main__":
    demo.launch(server_name='0.0.0.0', allowed_paths=[DEFAULT_VIDEO_DIR]) 
