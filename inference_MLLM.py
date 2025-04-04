import os
import hydra
from omegaconf import OmegaConf
import torch
import numpy as np
import re
import json
import pyrootutils
from PIL import Image
from tqdm import tqdm

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

# Number of image tokens
num_mm_tokens = 226

# Set random seed for reproducibility
np.random.seed(42)

# Device and data type settings for PyTorch
device = 'cuda:0'
dtype = torch.bfloat16
dtype_str = 'bf16'
is_video = False

# Number of tokens for video input and output
num_vid_in_tokens = 226
num_vid_out_tokens = 226

# Concatenate special tokens for processing
speacial_tokens = MM_BOI_TOKEN + BOI_TOKEN + \
    ''.join([FRAME_TOKEN.format(int(item)) \
    for item in range(num_vid_in_tokens)]) + \
    EOI_TOKEN + MM_EOI_TOKEN + WORLD_BML_TOKEN

# Template for instruction prompts
instruction_prompt = "{caption}"

# Configuration paths for tokenizer and models
tokenizer_cfg_path = './MLLM/configs/tokenizer/tokenizer.yaml'
llm_cfg_path = './MLLM/configs/clm_models/mistral-7b.yaml'
agent_cfg_path = './MLLM/configs/clm_models/animegamer_clm.yaml'

# Directory for saving results
save_dir = './results/multimodal_representations'
os.makedirs(save_dir, exist_ok=True)

# Simulation parameters
max_window = 1
step_by_step = True 

# Load configurations and instantiate models
tokenizer_cfg = OmegaConf.load(tokenizer_cfg_path)
tokenizer = hydra.utils.instantiate(tokenizer_cfg)

llm_cfg = OmegaConf.load(llm_cfg_path)
llm = hydra.utils.instantiate(llm_cfg, torch_dtype=dtype)
print('Init llm done.')

agent_model_cfg = OmegaConf.load(agent_cfg_path)
agent_model = hydra.utils.instantiate(agent_model_cfg, llm=llm)

agent_model.eval().to(device, dtype=dtype)
print('Init agent model Done')

# Encode the beginning of interaction token
boi_token_id = tokenizer.encode(BOI_TOKEN, add_special_tokens=False)[0]
ret = {}
count = 0


def format_world_strength(index):
    return STTOKEN.format(character_states[gen_turn_id][index])

# Load instructions from file
file_path = './game_demo/demo.txt'
instructions = []
with open(file_path, 'r', encoding='utf-8') as file:
    for line in file:
        instructions.append(line.strip())
    total_turn = len(instructions)
    # Initialize character states and embeddings
    character_states = [[0, 0, 0, 0] for i in range(total_turn)]
    mix_emb = torch.zeros([1, total_turn, 226, 1920]).to(device).to(dtype)
    max_repeat = 1

    for repeat in range(max_repeat):
        for gen_turn_id in tqdm(range(total_turn)):

            input_ids = [tokenizer.bos_token_id]
            ids_gen_mask = [False]
            ids_cmp_mask = [False]
            attention_mask = [1]
            emb_start = 0
            emb_end = emb_start + 1

            # Process previous turns for context
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

                    caption = instructions[turn]
                    SC = character_states[turn][0]
                    ET = character_states[turn][1]
                    ST = character_states[turn][2]
                    ML = character_states[turn][3]
                    caption = WORLD_BTURN_TOKEN + caption
                    caption_ids = tokenizer.encode(caption, add_special_tokens=False)

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
                    world_ids = tokenizer.encode(world_tokens, add_special_tokens=False)
                    # Construct the video tokens string
                    frame_tokens = ''.join([FRAME_TOKEN.format(int(item)) for item in range(num_mm_tokens)])
                    video_tokens = MM_BOI_TOKEN + (BOI_TOKEN + frame_tokens + EOI_TOKEN) * 1 + MM_EOI_TOKEN
                    # Encode the video tokens
                    video_ids = tokenizer.encode(video_tokens, add_special_tokens=False)
                    # Combine input IDs
                    input_ids += caption_ids + video_ids + world_ids
                    # Update attention mask
                    attention_mask += [1] * len(caption_ids) + [1] * len(video_ids) + [1] * len(world_ids)
                    # Update generation mask
                    ids_gen_mask += [False] * len(caption_ids) + [False] * len(video_ids) + [False] * len(world_ids)
                    # Update comparison mask
                    video_cmp_mask = [False] + ([False] + [True] * num_mm_tokens + [False]) * 1 + [False]
                    ids_cmp_mask += [False] * len(caption_ids) + video_cmp_mask + [False] * len(world_ids)
            # Process the current turn
            caption = instructions[gen_turn_id]
            caption = WORLD_BTURN_TOKEN + caption
            caption_ids = tokenizer.encode(caption, add_special_tokens=False)
            world_tokens = WORLD_BML_TOKEN
            world_ids = tokenizer.encode(world_tokens, add_special_tokens=False)            
            frame_sequence = ''.join([FRAME_TOKEN.format(int(item)) for item in range(num_mm_tokens)])
            video_tokens = (
                MM_BOI_TOKEN +
                (BOI_TOKEN + frame_sequence + EOI_TOKEN) * 1 +
                MM_EOI_TOKEN
            )
            video_ids = tokenizer.encode(video_tokens, add_special_tokens=False)

            # Combine input IDs
            input_ids += caption_ids + video_ids + world_ids

            # Generate masks for generation
            caption_gen_mask = [False] * len(caption_ids)
            video_gen_mask = [False] + ([False] + [True] * num_mm_tokens + [False]) * 1 + [False]
            world_gen_mask = [False] * len(world_ids)
            ids_gen_mask += caption_gen_mask + video_gen_mask + world_gen_mask

            # Generate masks for comparison
            caption_cmp_mask = [False] * len(caption_ids)
            video_cmp_mask = [False] * len(video_ids)
            world_cmp_mask = [False] * len(world_ids)
            ids_cmp_mask += caption_cmp_mask + video_cmp_mask + world_cmp_mask

            # Update attention mask
            caption_attention = [1] * len(caption_ids)
            video_attention = [1] + ([1] + [-1] * num_mm_tokens + [1]) * 1 + [1]
            world_attention = [1] * len(world_ids)
            attention_mask += caption_attention + video_attention + world_attention

            assert len(input_ids) == len(ids_gen_mask) == len(ids_cmp_mask)
            ids_gen_mask = torch.tensor(ids_gen_mask).to(device, dtype=torch.bool).unsqueeze(0)
            attention_mask = torch.tensor(attention_mask).to(device, dtype=torch.long).unsqueeze(0)
            input_ids = torch.tensor(input_ids).to(device, dtype=torch.long).unsqueeze(0)
            ids_cmp_mask = torch.tensor(ids_cmp_mask).to(device, dtype=torch.bool).unsqueeze(0)

            # Generate outputs without updating model weights
            with torch.no_grad():
                
                mix_emb_now_input = mix_emb[:,emb_start:emb_end,:,:]
                if gen_turn_id < max_window:
                    gen_id_tensor = torch.tensor(gen_turn_id).unsqueeze(0)
                else:
                    gen_id_tensor = torch.tensor(max_window-1).unsqueeze(0)

                txt_output = agent_model.generate(tokenizer=tokenizer,
                                                  input_ids=input_ids,
                                                  video_embeds=mix_emb_now_input,
                                                  gen_turn_id=gen_id_tensor,
                                                  ids_cmp_mask=ids_cmp_mask,
                                                  max_new_tokens=200)['text']

                vid_output = agent_model.generate_bi(tokenizer=tokenizer, 
                                                     input_ids=input_ids, 
                                                     attention_mask=attention_mask, 
                                                     ids_gen_mask=ids_gen_mask, 
                                                     ids_cmp_mask=ids_cmp_mask,
                                                     gen_turn_id=gen_id_tensor,
                                                     num_vid_in_tokens=num_vid_in_tokens, 
                                                     video_embeds=mix_emb_now_input,
                                                     num_vid_out_tokens=num_vid_out_tokens, 
                                                     device=device)

            # Parse and store the predicted character states
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
            
            # Update the embeddings and character states if step-by-step is enabled
            if step_by_step:
                mix_emb[0][gen_turn_id] = vid_output
                character_states[gen_turn_id] = pred_character_states
            
            # Store the results for this turn
            ret[f"turn_{gen_turn_id+1}"] = [vid_output.cpu(), instructions[gen_turn_id], pred_character_states]
            # Clear the GPU cache to free up memory
            torch.cuda.empty_cache()

# Save the results to a file
torch.save(
    ret,
    os.path.join(save_dir, "mllm-output.pt"),
)
print('MLLM down')
