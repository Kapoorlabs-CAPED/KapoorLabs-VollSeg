#!/bin/bash
# U-Net optimizer + LR + scheduler sweep (Jean Zay).
#
# Sweeps 3 optimizers × 3 LRs × 2 schedulers = 18 runs as a SLURM job
# array. Same resources as the single-shot training script.
#
# LR ladders:
#   adam → 1.0e-3 .. 1.0e-1
#   sgd  → 1.0e-1 .. 1.0e+1
#   lars → 1.0e-1 .. 1.0e+1
#
# Submit: sbatch slurm_sweep_unet_jeanzay.sh

#SBATCH --nodes=1
#SBATCH -A lzc@a100
#SBATCH -C a100
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_p5
#SBATCH --job-name=unet_sweep
#SBATCH --cpus-per-task=40
#SBATCH --time=20:00:00
#SBATCH --array=0-17%6
#SBATCH --output=sweep_logs/unet_sweep_%A_%a.o
#SBATCH --error=sweep_logs/unet_sweep_%A_%a.e

#SBATCH --requeue
#SBATCH --signal=SIGTERM@180

mkdir -p sweep_logs

echo "running in shell: $SHELL"
echo "array task id: $SLURM_ARRAY_TASK_ID"
export NCCL_SOCKET_IFNAME=lo

module purge
module load anaconda-py3
conda deactivate
conda activate torchenv
module load cuda/11.8.0

OPTIMIZERS=(adam sgd lars)
ADAM_LRS=(1.0e-3 1.0e-2 1.0e-1)
SGD_LRS=(1.0e-1 1.0e+0 1.0e+1)
LARS_LRS=(1.0e-1 1.0e+0 1.0e+1)
SCHEDULERS=(null cosine)

TASK=${SLURM_ARRAY_TASK_ID}
sched_idx=$((TASK % 2))
TASK=$((TASK / 2))
lr_idx=$((TASK % 3))
TASK=$((TASK / 3))
opt_idx=$((TASK % 3))

OPTIMIZER=${OPTIMIZERS[$opt_idx]}
SCHEDULER=${SCHEDULERS[$sched_idx]}

case "$OPTIMIZER" in
  adam) LR=${ADAM_LRS[$lr_idx]} ;;
  sgd)  LR=${SGD_LRS[$lr_idx]}  ;;
  lars) LR=${LARS_LRS[$lr_idx]} ;;
esac

LR_TAG=$(echo "$LR" | sed 's/[.+]/p/g')
SCHED_TAG=$([ "$SCHEDULER" = "null" ] && echo "noscheduler" || echo "$SCHEDULER")
EXPERIMENT="unet_sweep_${OPTIMIZER}_lr${LR_TAG}_${SCHED_TAG}"

echo "─────────────────────────────────────────────────"
echo "  optimizer = $OPTIMIZER"
echo "  lr        = $LR"
echo "  scheduler = $SCHEDULER"
echo "  experiment_name = $EXPERIMENT"
echo "─────────────────────────────────────────────────"

srun --unbuffered python lightning-unet.py \
    train_data_paths=xenopus_jeanzay \
    train_data_paths.experiment_name="$EXPERIMENT" \
    train_data_paths.log_path="/lustre/fsn1/projects/rech/jsy/uzj81mi/models_unet_pytorch_sweep/$EXPERIMENT/" \
    parameters.optimizer="$OPTIMIZER" \
    parameters.learning_rate="$LR" \
    parameters.scheduler="$SCHEDULER" \
    parameters.scheduler_kwargs.t_max=200 \
    parameters.scheduler_kwargs.eta_min=1.0e-6
