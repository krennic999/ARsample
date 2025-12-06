import os
import sys
sys.path.append("./")

import torch
from torchvision import transforms
import random
from src.transformer import Transformer2DModel
from src.pipeline import Pipeline
# from src.pipeline_motivation import Pipeline
from src.scheduler import Scheduler
from transformers import (
    CLIPTextModelWithProjection,
    CLIPTokenizer,
)
from diffusers import VQModel

seed=4000
torch.manual_seed(seed)
random.seed(seed)

device = 'cuda'
dtype = torch.bfloat16
model_path = "/your-model-path"
model = Transformer2DModel.from_pretrained(model_path, subfolder="transformer", torch_dtype=dtype)
vq_model = VQModel.from_pretrained(model_path, subfolder="vqvae", torch_dtype=dtype)
# text_encoder = CLIPTextModelWithProjection.from_pretrained(model_path,subfolder="text_encoder", torch_dtype=dtype)
text_encoder = CLIPTextModelWithProjection.from_pretrained(   #using original text enc for stable sampling
                "/your-ckpt/ckpts/laion/CLIP-ViT-H-14-laion2B-s32B-b79K",torch_dtype=dtype)
tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer", torch_dtype=dtype)
scheduler = Scheduler.from_pretrained(model_path, subfolder="scheduler")
pipe=Pipeline(vq_model, tokenizer=tokenizer,text_encoder=text_encoder,transformer=model,scheduler=scheduler)
pipe = pipe.to(device)

pipe = pipe.to(device)

steps = 64
CFG = 1.0
resolution = 1024 
negative_prompt = "worst quality, low quality, low res, blurry, distortion, watermark, logo, signature, text, jpeg artifacts, signature, sketch, duplicate, ugly, identifying mark"

# prompts = [
#     "Two actors are posing for a pictur with one wearing a black and white face paint.",
# ]
prompts = ["A snowy owl is sitting in the snow"]*12

batched_generation = True
num_images = len(prompts) if batched_generation else 1

images = pipe(
    prompt=prompts[:num_images], 
    negative_prompt=[negative_prompt] * num_images,
    height=resolution,
    width=resolution,
    guidance_scale=CFG,
    top_k=2000,
    num_inference_steps=steps,
    enable_entropy_filtering=False,
    # entropy_range=[[0,100]],
    # temperature_value=[1.5]
    ).images

output_dir = "./output3"
os.makedirs(output_dir, exist_ok=True)
for i, prompt in enumerate(prompts[:num_images]):
    sanitized_prompt = prompt.replace(" ", "_")
    file_path = os.path.join(output_dir, f"{sanitized_prompt}_{resolution}_{steps}_{CFG}_{i}.png")
    images[i].save(file_path)
    print(f"The {i+1}/{num_images} image is saved to {file_path}")
