# !/bin/bash
set -x

module load compiler/gcc/gcc-11.3.0-gcc-4.8.5-exnyzqi
# bash scripts/autoregressive/train_c2i.sh --cloud-save-path /path/to/cloud_disk 
# --code-path /path/to/imagenet_code_c2i_flip_ten_crop --image-size 384 --gpt-model GPT-XL

torchrun \
--nnodes=1 --nproc_per_node=1 --node_rank=0 \
--master_addr=127.0.0.1 --master_port=15666 \
autoregressive/train/train_verifier.py

# torchrun \
# --nnodes=1 --nproc_per_node=1 --node_rank=0 \
# --master_addr=127.0.0.1 --master_port=15666 \
# autoregressive/train/valid_verifier.py
