import torch
from transformers import LogitsProcessor

# Define special tokens for image and video processing
IMG_BOI_TOKEN = "<img>"
IMG_EOI_TOKEN = "</img>"
VID_BOI_TOKEN = "<vid>"
VID_EOI_TOKEN = "</vid>"
BOI_TOKEN = "<frame>"
EOI_TOKEN = "</frame>"
FRAME_TOKEN = '<frame_{:05d}>'  # Frame token with zero-padded numbering

# Processor class for generating video tokens automatically
class AutoVideoTokenGenerationProcessor(LogitsProcessor):
    def __init__(self, tokenizer, num_vid_gen_tokens=100) -> None:
        super().__init__()
        # Create a string of video tokens from start to end
        vid_all_token_str = ''.join([BOI_TOKEN] + [FRAME_TOKEN.format(int(item))
                                                   for item in range(num_vid_gen_tokens)] + [EOI_TOKEN])
        # Encode the string of tokens into token IDs
        self.vid_ids_list = tokenizer.encode(vid_all_token_str, add_special_tokens=False)

    def __call__(self, input_ids, scores):
        bz = input_ids.shape[0]  # Batch size
        for i in range(bz):
            cur_input_id = input_ids[i, -1].item()  # Get the last token ID in the sequence
            if cur_input_id in self.vid_ids_list[:-1]:  # Check if it's not the last token
                # Get the next token ID in the sequence
                output_id = self.vid_ids_list[self.vid_ids_list.index(cur_input_id) + 1]
                # Boost the score of the next token to make it more likely to be chosen
                scores[i, ..., output_id] = scores[i, ...].max() + 10.
            else:
                # Set scores of all other tokens to zero to prevent them from being chosen
                scores[i, ..., torch.tensor(self.vid_ids_list[1:]).to(dtype=torch.long)] = 0.0

        return scores
    
# Processor class for generating image tokens automatically
class AutoImageTokenGenerationProcessor(LogitsProcessor):
    def __init__(self, tokenizer, num_img_gen_tokens=64) -> None:
        super().__init__()
        # Create a string of image tokens from start to end
        img_all_token_str = ''.join([BOI_TOKEN] + [IMG_TOKEN.format(int(item))
                                                   for item in range(num_img_gen_tokens)] + [EOI_TOKEN])
        # Encode the string of tokens into token IDs
        self.img_ids_list = tokenizer.encode(img_all_token_str, add_special_tokens=False)


    def __call__(self, input_ids, scores):
        bz = input_ids.shape[0]  # Batch size
        for i in range(bz):
            cur_input_id = input_ids[i, -1].item()  # Get the last token ID in the sequence
            if cur_input_id in self.img_ids_list[:-1]:  # Check if it's not the last token
                # Get the next token ID in the sequence
                output_id = self.img_ids_list[self.img_ids_list.index(cur_input_id) + 1]
                # Boost the score of the next token to make it more likely to be chosen
                scores[i, ..., output_id] = scores[i, ...].max() + 10.
            else:
                # Set scores of all other tokens to zero to prevent them from being chosen
                scores[i, ..., torch.tensor(self.img_ids_list[1:]).to(dtype=torch.long)] = 0.0

        return scores