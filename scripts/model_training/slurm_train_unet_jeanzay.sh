#!/bin/bash
#SBATCH --nodes=1
#SBATCH -A lzc@a100
#SBATCH -C a100
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_p5
#SBATCH --job-name=unet
#SBATCH --cpus-per-task=40
#SBATCH --output=unet_train.o%j
#SBATCH --error=unet_train.o%j
#SBATCH --time=20:00:00

#SBATCH --requeue
#SBATCH --signal=SIGTERM@180
echo "running in shell: " "$SHELL"
export NCCL_SOCKET_IFNAME=lo

module purge
module load anaconda-py3
conda deactivate
conda activate torchenv
module load cuda/11.8.0

srun --unbuffered python lightning-unet.py \
    train_data_paths=xenopus_jeanzay
