# AnimeGamer: Infinite Anime Life Simulation with Next Game State Prediction

**[Junhao Cheng<sup>1,2</sup>](https://donahowe.github.io/), 
[Yuying Ge<sup>1,&#9993;</sup>](https://geyuying.github.io/), 
[Yixiao Ge<sup>1</sup>](https://geyixiao.com/), 
[Jing Liao<sup>2</sup>](https://scholar.google.com/citations?user=3s9f9VIAAAAJ&hl=zh-CN), 
[Ying Shan<sup>1</sup>](https://scholar.google.com/citations?user=4oXBp9UAAAAJ&hl=en)**
<br>
<sup>1</sup>ARC Lab, Tencent PCG, 
<sup>2</sup>City University of Hong Kong
<br>



## 🔎 Introduction

![teaser](assets/Intro2.gif)
![teaser](assets/Intro.gif)

We propose AnimeGamer for infinite anime life simulation. AnimeGamer is built upon Multimodal Large Language Models (MLLMs) to generate each game state, including dynamic animation shots that depict character movements and updates to character states. The overview of AnimeGamer is as follows. The training process consists of three phases:
* (a) We model animation shots using action-aware multimodal representations through an encoder and train a diffusion-based decoder to reconstruct videos, with the additional input of motion scope that indicates action intensity. 
* (b) We train an MLLM to predict the next game state representations by taking the history instructions and game state representations as input.
* (c) We further enhance the quality of decoded animation shots from the MLLM via an adaptation phase, where the decoder is fine-tuned by taking MLLM's predictions as input.

![teaser](assets/model.png)



## 📅 News

* [2025-03-28] Create the repository. 🔥🔥🔥


## 🔜 TODOs
- [ ] Release training codes 
- [ ] Release inference codes 
- [ ] release wights of models trained on "Qiqi's Delivery Service" and "Ponyo on the Cliff" seperately. 

## 📏 Inference

Please first download the checkpoints of [AnimeGamer](https://huggingface.co/msj9817/GenHancer/) and [Mistral-7B](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1), and save them under the folder `./checkpoints`.

To generate action-aware multimodal representations and update character states, you can run:
```shell
python inference_MLLM.py 
```

To decode the representations into animation shots, you can run:
```shell
python inference_Decoder.py 
```

Change the instructions in `./game_demo` to customize your play.



## 🤗 Acknowledgements

We refer to [CogvideoX](https://github.com/XLabs-AI/x-flux) and [SEED-X](https://github.com/AILab-CVC/SEED-X/tree/main) to build our codebase. Thanks for their wonderful project.
