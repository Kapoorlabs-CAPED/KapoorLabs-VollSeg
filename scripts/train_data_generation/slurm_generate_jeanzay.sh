#!/bin/bash
#SBATCH --nodes=1
#SBATCH -A lzc@cpu
#SBATCH --partition=cpu_p1
#SBATCH --job-name=vs_gen
#SBATCH --cpus-per-task=20
#SBATCH --output=gen.o%j
#SBATCH --error=gen.o%j
#SBATCH --time=10:00:00

module purge
module load anaconda-py3
conda deactivate
conda activate torchenv

srun --unbuffered python generate-training-data.py \
    train_data_paths=xenopus_jeanzay
