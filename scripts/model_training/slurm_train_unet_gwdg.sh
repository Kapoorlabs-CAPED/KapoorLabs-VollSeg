#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --job-name=vs_train_unet
#SBATCH --output=unet_train_%j.out
#SBATCH --error=unet_train_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:A100:1
#SBATCH --partition=grete:shared
#SBATCH --mem=32G

#SBATCH --requeue
#SBATCH --signal=SIGTERM@180
echo "running in shell: " "$SHELL"
export NCCL_SOCKET_IFNAME=lo

module purge
module load cuda
module load miniforge3
source activate torchenv

srun --unbuffered python lightning-unet.py \
    train_data_paths=xenopus_gwdg
