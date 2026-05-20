#!/bin/bash
# CARE optimizer + LR + scheduler sweep (Jean Zay).
#
# Sweeps 3 optimizers × 3 LRs × 2 schedulers = 18 runs as a SLURM job
# array. Same resources as the single-shot training script.
#
# LR ladders:
#   adam → 1.0e-3 .. 1.0e-1
#   sgd  → 1.0e-1 .. 1.0e+1
#   lars → 1.0e-1 .. 1.0e+1
#
# Submit: sbatch slurm_sweep_care_jeanzay.sh

#SBATCH --nodes=1
#SBATCH -A lzc@a100
#SBATCH -C a100
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_p5
#SBATCH --job-name=care_sweep
#SBATCH --cpus-per-task=40
#SBATCH --time=20:00:00
#SBATCH --array=0-17%6
#SBATCH --output=sweep_logs/care_sweep_%A_%a.o
#SBATCH --error=sweep_logs/care_sweep_%A_%a.e

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
SCHEDULERS=(none cosine)

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
EXPERIMENT="care_sweep_${OPTIMIZER}_lr${LR_TAG}_${SCHEDULER}"

# CARE uses the kietzmann-style ``_target_`` hydra-instantiate pattern
# for its scheduler. Map the sweep's string-keyed choice onto the right
# class path so the existing lightning-care.py script picks it up
# unchanged.
case "$SCHEDULER" in
  none|noscheduler)
    SCHED_TARGET="kapoorlabs_vollseg.care_lightning.schedulers.SameLR"
    ;;
  cosine)
    SCHED_TARGET="kapoorlabs_vollseg.care_lightning.schedulers.CosineAnnealingScheduler"
    ;;
esac

echo "─────────────────────────────────────────────────"
echo "  optimizer = $OPTIMIZER"
echo "  lr        = $LR"
echo "  scheduler = $SCHEDULER ($SCHED_TARGET)"
echo "  experiment_name = $EXPERIMENT"
echo "─────────────────────────────────────────────────"

srun --unbuffered python lightning-care.py \
    train_data_paths=care_jeanzay \
    train_data_paths.experiment_name="$EXPERIMENT" \
    parameters.optimizer="$OPTIMIZER" \
    parameters.learning_rate="$LR" \
    "parameters.scheduler._target_=$SCHED_TARGET"
