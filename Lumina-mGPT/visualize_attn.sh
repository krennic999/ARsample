#!/bin/bash
#SBATCH --job-name=download_model    # 作业名称
#SBATCH --output=download_model_%j.log  # 输出日志文件
#SBATCH -N 1
#SBATCH --ntasks=1                # 任务数
#SBATCH --cpus-per-task=1         # 每个任务的CPU核数
#SBATCH --mem=4G                  # 任务分配的内存
#SBATCH --time=00:30:00            # 预计任务运行时间    

# module load python/3.8

# source /path/to/your/venv/bin/activate

# apptainer shell --bind /mnt/petrelfs/maxiaoxiao:/mnt/petrelfs/maxiaoxiao /mnt/petrelfs/maxiaoxiao/ubuntu20.04-py3.8-cuda11.8-cudnn8.9-torch2.0-deepspeed0.9.5_v1.0.0.sif
# python /mnt/petrelfs/maxiaoxiao/download_data.py --repo_id Alpha-VLLM/Lumina-mGPT-7B-1024 --download_dir /mnt/petrelfs/maxiaoxiao/Lumina-mGPT-7B-1024

apptainer exec --bind /mnt/petrelfs/maxiaoxiao:/mnt/petrelfs/maxiaoxiao --nv \
  /mnt/petrelfs/maxiaoxiao/ubuntu20.04-py3.8-cuda11.8-cudnn8.9-torch2.0-deepspeed0.9.5_v1.0.0.sif \
  python /mnt/petrelfs/maxiaoxiao/codes/ar_sampling/Lumina-mGPT/lumina_mgpt/visualize_attn.py