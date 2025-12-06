import os
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
from transformers import AutoTokenizer
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import time
import argparse
from collections import OrderedDict
import random
import pdb
import json
import torch
from torchvision import transforms
from src.transformer import Transformer2DModel
from src.pipeline import Pipeline
from src.scheduler import Scheduler
from transformers import (
    CLIPTextModelWithProjection,
    CLIPTokenizer,
)
from diffusers import VQModel

def normalize_01_into_pm1(x):  # normalize x from [0, 1] to [-1, 1] by (x*2) - 1
    return x.add(x).add_(-1)

def transform_image(image, size=512):
    transform = transforms.Compose([
        transforms.Resize(size, max_size=None),  # 等比缩放，最小边对齐 size
        transforms.CenterCrop(size),  # 居中裁剪成 size x size
        transforms.ToTensor(),
        normalize_01_into_pm1,
    ])
    return transform(image)
negative_prompt = "worst quality, low quality, low res, blurry, distortion, watermark, logo, signature, text, jpeg artifacts, signature, sketch, duplicate, ugly, identifying mark"

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

def setup(rank, world_size):
    """初始化分布式环境"""
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29501'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    """销毁分布式进程"""
    dist.destroy_process_group()
    
def deduplicate_annotations(meta_json):
    unique_annotations = {}
    for annotation in meta_json['annotations']:
        unique_annotations[annotation['image_id']] = annotation  # 覆盖旧的，保留最新的
    meta_json['annotations'] = list(unique_annotations.values())

def get_prompt_data_for_rank(prompts_dir, rank, world_size, batch_size):
    # 收集所有 prompt 条目
    all_prompts=[]
    all_prompt_files=[]
    
    prompt_files = sorted([f for f in os.listdir(prompts_dir) if f.endswith('.txt')])
    for idx, prompt_file in enumerate(prompt_files):
        with open(os.path.join(prompts_dir, prompt_file), 'r') as f:
            prompt = f.readline().strip()
            all_prompts.append(prompt)
            all_prompt_files.append(prompt_file.split('.')[0])

    total_samples = len(all_prompts)
    if rank == 0:
        print(f"Total prompt samples: {total_samples}")

    # 分配数据到每个 rank
    samples_per_rank = total_samples // world_size
    start_idx = rank * samples_per_rank
    end_idx = start_idx + samples_per_rank if rank != world_size - 1 else total_samples
    rank_items = all_prompts[start_idx:end_idx]
    rank_files=all_prompt_files[start_idx:end_idx]

    # 按 batch 分组，返回格式为 dict of lists
    batched_data = []
    for i in range(0, len(rank_items), batch_size):
        batch = rank_items[i:i+batch_size]
        batch_files=rank_files[i:i+batch_size]
        batch_dict = {
            'caption': batch,
            'name': batch_files,
        }
        batched_data.append(batch_dict)

    return batched_data
    
    
def remove_module_prefix(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_key = k.replace("module.", "")
        new_state_dict[new_key] = v
    return new_state_dict



# Function to save the grid image
def save_image_grid(images, save_path, grid_size=(2, 2), img_size=(1024, 1024)):
    grid_img = Image.new('RGB', (grid_size[1] * img_size[1], grid_size[0] * img_size[0]))
    for idx, img in enumerate(images):
        x = idx % grid_size[1] * img_size[1]
        y = idx // grid_size[1] * img_size[0]
        grid_img.paste(img, (x, y))
    grid_img.save(save_path)


def generate_images_for_prompt(args, text_prompts, model, num_images=4, seed=None, enable_entropy_filtering=False, rank=0):
    if seed:
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)  # 多GPU时
    
    with torch.no_grad():
            samples = model(
                prompt=[text_prompts] * num_images, 
                negative_prompt=[negative_prompt] * num_images,
                height=1024,
                width=1024,
                guidance_scale=9,
                enable_entropy_filtering=enable_entropy_filtering,
                num_inference_steps=64
                ).images
            
    return samples


def main(args):
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    seed=20
    depth=30
    batch_size = 1 # 每个 GPU 处理 batch_size=8，总的 batch_size = 8 * world_size
    enable_entropy_filtering=args.enable_entropy_filtering=='True'
    
    setup(rank, world_size)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_grad_enabled(False)
    # 如果用的是 GPU，还要加上：
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  # 多GPU时
    
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
    if rank == 0:
        print("VQ model loaded")

    prompts_dir = '/your-dpg-prompt-root/dpg_bench/prompts'
    batched_data=get_prompt_data_for_rank(prompts_dir,rank,world_size,batch_size)
    print('enable_entropy_filtering: ',enable_entropy_filtering)
    save_root=args.save_root+'/entropy%s_seed%d'%(args.enable_entropy_filtering,args.seed)
    os.makedirs(save_root,exist_ok=True)

    # 处理数据集
    for item in batched_data:
        t2=time.time()
        print(item)
        text_prompts = item['caption']
        prompt_file=item['name']
        print('rank %d: '%rank,type)
        B_=len(text_prompts)
        print(text_prompts)

        images = generate_images_for_prompt(args, text_prompts[0], pipe, num_images=4, seed=args.seed, enable_entropy_filtering=args.enable_entropy_filtering, rank=rank)
        
        output_path = os.path.join(save_root, f"{prompt_file[0]}.png")
        save_image_grid(images, output_path,img_size=(args.image_size,args.image_size))
        print(f"Saved: {output_path}")

    cleanup()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--precision", type=str, default='bf16', choices=["none", "fp16", "bf16"]) 
    parser.add_argument("--compile", action='store_true', default=False)
    parser.add_argument("--codebook-size", type=int, default=16384, help="codebook size for vector quantization")
    parser.add_argument("--codebook-embed-dim", type=int, default=8, help="codebook dimension for vector quantization")
    parser.add_argument("--image-size", type=int, choices=[256, 384, 512, 1024], default=1024)
    parser.add_argument("--downsample-size", type=int, choices=[8, 16], default=16)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=0, help="top-k value to sample with")
    parser.add_argument("--enable_entropy_filtering", type=str, default='True', help="entropy to sample with")
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature value to sample with")
    parser.add_argument("--top-p", type=float, default=1.0, help="top-p value to sample with")
    parser.add_argument("--save_root", type=str, default='/your-save-root/seed_test_dpg/')
    args = parser.parse_args()
    main(args)