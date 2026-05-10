#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --job-name=vs_gen
#SBATCH --output=gen_%j.out
#SBATCH --error=gen_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=medium
#SBATCH --mem=32G

module purge
module load miniforge3
source activate torchenv

srun --unbuffered python generate-training-data.py \
    train_data_paths=xenopus_gwdg
