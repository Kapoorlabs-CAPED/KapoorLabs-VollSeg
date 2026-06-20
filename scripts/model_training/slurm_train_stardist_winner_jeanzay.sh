#!/bin/bash
# Retrain the StarDist sweep winner with anisotropy matching the keras
# Xenopus models (Z:Y:X = 17/7 : 1 : 1 ≈ 2.4286 : 1 : 1). The original
# sweep used the prior default anisotropy = (2, 1, 1) which produced
# systematically smaller cells (lower volume / surface area / radius)
# than the keras reference because every Z-ray was scaled too short.
#
# Winner hyperparameters from sweep_predict_summary.csv:
#   optimizer = adam
#   lr        = 1.0e-3
#   scheduler = noscheduler (null)
#
# Submit: sbatch slurm_train_stardist_winner_jeanzay.sh

#SBATCH --nodes=1
#SBATCH -A lzc@a100
#SBATCH -C a100
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_p5
#SBATCH --job-name=stardist_winner
#SBATCH --cpus-per-task=40
#SBATCH --time=20:00:00
#SBATCH --output=stardist_winner_%j.o
#SBATCH --error=stardist_winner_%j.e

#SBATCH --requeue
#SBATCH --signal=SIGTERM@180

echo "running in shell: $SHELL"
export NCCL_SOCKET_IFNAME=lo

module purge
module load anaconda-py3
conda deactivate
conda activate torchenv
module load cuda/11.8.0

# Promote folder lives outside ``models_stardist_pytorch_sweep`` so it
# isn't picked up by the next sweep-predict run. Threshold optimiser
# and HF upload should both point at this path after training finishes.
LOG_PATH="/lustre/fsn1/projects/rech/jsy/uzj81mi/models_stardist_pytorch/"
EXPERIMENT="xenopus_stardist_winner_anisotropic"

mkdir -p "$LOG_PATH"

# parameters.anisotropy is read from the yaml's new default
# (2.4285714... = 17/7), so we don't pass it explicitly — change it on
# the CLI only if you want a different Z:XY ratio for a re-run.
srun --unbuffered python lightning-stardist.py \
    train_data_paths=xenopus_jeanzay \
    train_data_paths.experiment_name="$EXPERIMENT" \
    train_data_paths.log_path="$LOG_PATH" \
    parameters.optimizer=adam \
    parameters.learning_rate=1.0e-3 \
    parameters.scheduler=null
