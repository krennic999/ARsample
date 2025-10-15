import sys
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
from tokenizer.tokenizer_image.vq_model import VQ_models
from language.t5 import T5Embedder
from autoregressive.models.gpt import GPT_models
from autoregressive.models.generate import generate
from transformers import (
    CLIPTextModelWithProjection,
    CLIPTokenizer,
)
from diffusers import VQModel

def normalize_01_into_pm1(x):  # normalize x from [0, 1] to [-1, 1] by (x*2) - 1
    return x.add(x).add_(-1)

def transform_image(image, size=512):
    transform = transforms.Compose([
        transforms.Resize(size, max_size=None),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        normalize_01_into_pm1,
    ])
    return transform(image)

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29502'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()
    
def deduplicate_annotations(meta_json):
    unique_annotations = {}
    for annotation in meta_json['annotations']:
        unique_annotations[annotation['image_id']] = annotation  # 覆盖旧的，保留最新的
    meta_json['annotations'] = list(unique_annotations.values())

def get_prompt_data_for_rank(prompts_dir, rank, world_size, batch_size):
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

    samples_per_rank = total_samples // world_size
    start_idx = rank * samples_per_rank
    end_idx = start_idx + samples_per_rank if rank != world_size - 1 else total_samples
    rank_items = all_prompts[start_idx:end_idx]
    rank_files=all_prompt_files[start_idx:end_idx]

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


def generate_images_for_prompt(args, text_prompts, gpt_model, t5_model, vq_model, num_images=4, seed=None, enable_entropy_filtering=False, rank=0):
    if seed:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    images = []
    latent_size = args.image_size // args.downsample_size
    with torch.no_grad():
        caption_embs, emb_masks = t5_model.get_text_embeddings([text_prompts]*num_images)

        if not args.no_left_padding:
            new_emb_masks = torch.flip(emb_masks, dims=[-1])
            new_caption_embs = []
            for caption_emb, emb_mask in zip(caption_embs, emb_masks):
                valid_num = int(emb_mask.sum().item())
                new_caption_emb = torch.cat([caption_emb[valid_num:], caption_emb[:valid_num]])
                new_caption_embs.append(new_caption_emb)
            new_caption_embs = torch.stack(new_caption_embs)
        else:
            new_caption_embs, new_emb_masks = caption_embs, emb_masks

        c_indices = new_caption_embs * new_emb_masks[:, :, None]
        c_emb_masks = new_emb_masks

        qzshape = [len(c_indices), args.codebook_embed_dim, latent_size, latent_size]
        t1 = time.time()
        index_sample, entropy_list = generate(
            gpt_model.module, c_indices, latent_size ** 2,
            c_emb_masks,
            cfg_scale=args.cfg_scale,
            temperature=args.temperature, top_k=args.top_k,
            top_p=args.top_p, enable_entropy_filtering=args.enable_entropy_filtering, sample_logits=True
        )
        if rank == 0:
            print(f"Full sampling takes about {time.time() - t1:.2f} seconds.")

        t2 = time.time()
        samples = vq_model.module.decode_code(index_sample, qzshape)
        if rank == 0:
            print(f"Decoder takes about {time.time() - t2:.2f} seconds.")
            
        for i in range(num_images):
            img_pred = (samples[i].permute(1, 2, 0).add_(1).mul_(0.5).clamp_(0, 1).cpu() * 255.).numpy().astype(np.uint8)
            images.append(Image.fromarray(img_pred))
    return images


def main(args):
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    seed=20
    depth=30
    batch_size = 1 
    enable_entropy_filtering=args.enable_entropy_filtering
    
    setup(rank, world_size)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_grad_enabled(False)
    
    device = 'cuda'
    vq_model = VQ_models[args.vq_model](
        codebook_size=args.codebook_size,
        codebook_embed_dim=args.codebook_embed_dim
    ).to(device)
    vq_model.load_state_dict(remove_module_prefix(torch.load(args.vq_ckpt, map_location="cpu")["model"]), strict=False)
    vq_model.eval()
    vq_model = DDP(vq_model, device_ids=[rank], output_device=rank)
    if rank == 0:
        print("VQ model loaded")

    precision = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.precision]
    latent_size = args.image_size // args.downsample_size
    gpt_model = GPT_models[args.gpt_model](
        block_size=latent_size ** 2,
        cls_token_num=args.cls_token_num,
        model_type=args.gpt_type,
        require_lora=False
    ).to(device=device, dtype=precision)

    gpt_model.load_state_dict(remove_module_prefix(torch.load(args.gpt_ckpt, map_location="cpu")["model"]), strict=False)
    gpt_model.eval()
    gpt_model = DDP(gpt_model, device_ids=[rank], output_device=rank)
    if rank == 0:
        print("GPT model loaded")
        
    t5_model = T5Embedder(
        device=device,
        local_cache=True,
        cache_dir=args.t5_path,
        dir_or_name=args.t5_model_type,
        torch_dtype=precision,
        model_max_length=args.t5_feature_max_len,
    )

    prompts_dir = 'your_dpg/dpg_bench/prompts'
    batched_data=get_prompt_data_for_rank(prompts_dir,rank,world_size,batch_size)
    print('enable_entropy_filtering: ',enable_entropy_filtering)
    os.makedirs(args.save_root,exist_ok=True)

    for item in batched_data:
        t2=time.time()
        print(item)
        text_prompts = item['caption']
        prompt_file=item['name']
        print('rank %d: '%rank,type)
        B_=len(text_prompts)
        print(text_prompts)
        images = generate_images_for_prompt(args, text_prompts[0], gpt_model, t5_model, vq_model, num_images=4, seed=args.seed, enable_entropy_filtering=args.enable_entropy_filtering, rank=rank)
        
        output_path = os.path.join(args.save_root, f"{prompt_file[0]}.png")
        save_image_grid(images, output_path,img_size=(args.image_size,args.image_size))
        print(f"Saved: {output_path}")

    cleanup()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--t5-path", type=str, default='your_t5/google/')
    parser.add_argument("--t5-model-type", type=str, default='flan-t5-xl')
    parser.add_argument("--t5-feature-max-len", type=int, default=120)
    parser.add_argument("--t5-feature-dim", type=int, default=2048)
    parser.add_argument("--no-left-padding", action='store_true', default=False)
    parser.add_argument("--gpt-model", type=str, choices=list(GPT_models.keys()), default="GPT-XL")
    parser.add_argument("--gpt-ckpt", type=str, default='your_gpt/t2i_XL_stage1_256.pt')
    parser.add_argument("--gpt-type", type=str, choices=['c2i', 't2i'], default="t2i", help="class->image or text->image")  
    parser.add_argument("--cls-token-num", type=int, default=120, help="max token number of condition input")
    parser.add_argument("--precision", type=str, default='bf16', choices=["none", "fp16", "bf16"]) 
    parser.add_argument("--compile", action='store_true', default=False)
    parser.add_argument("--vq-model", type=str, choices=list(VQ_models.keys()), default="VQ-16")
    parser.add_argument("--vq-ckpt", type=str, default='your_vq/vq_ds16_t2i.pt', help="ckpt path for vq model")
    parser.add_argument("--codebook-size", type=int, default=16384, help="codebook size for vector quantization")
    parser.add_argument("--codebook-embed-dim", type=int, default=8, help="codebook dimension for vector quantization")
    parser.add_argument("--image-size", type=int, choices=[256, 384, 512], default=256)
    parser.add_argument("--downsample-size", type=int, choices=[8, 16], default=16)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=0, help="top-k value to sample with")
    parser.add_argument("--enable_entropy_filtering", type=bool, default=True, help="entropy to sample with")
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature value to sample with")
    parser.add_argument("--top-p", type=float, default=1.0, help="top-p value to sample with")
    parser.add_argument("--save_root", type=str, default='your_save_path')
    args = parser.parse_args()
    main(args)