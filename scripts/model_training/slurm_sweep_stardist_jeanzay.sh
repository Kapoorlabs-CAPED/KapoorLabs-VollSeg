#!/bin/bash
# StarDist optimizer + LR + scheduler sweep (Jean Zay).
#
# Sweeps 3 optimizers × 3 LRs × 2 schedulers = 18 runs as a SLURM job
# array. Same node / GPU / account / walltime as the single-shot
# training script.
#
# LR ladders (per user spec):
#   adam → 1.0e-3 .. 1.0e-1   (low side; typical Adam range)
#   sgd  → 1.0e-1 .. 1.0e+1   (high side; typical SGD range)
#   lars → 1.0e-1 .. 1.0e+1   (high side; LARS is SGD-like)
#
# Submit:
#   sbatch slurm_sweep_stardist_jeanzay.sh
# Each task writes to log_path/<experiment_name>_<opt>_<lr>_<sched>/
# so checkpoints don't collide.

#SBATCH --nodes=1
#SBATCH -A lzc@a100
#SBATCH -C a100
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_p5
#SBATCH --job-name=stardist_sweep
#SBATCH --cpus-per-task=40
#SBATCH --time=20:00:00
#SBATCH --array=0-17%6
#SBATCH --output=sweep_logs/stardist_sweep_%A_%a.o
#SBATCH --error=sweep_logs/stardist_sweep_%A_%a.e

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

# ── sweep grid ─────────────────────────────────────────────────
# task_id = ((opt_idx) * 3 + lr_idx) * 2 + sched_idx
#   opt_idx   ∈ {0:adam, 1:sgd, 2:lars}
#   lr_idx    ∈ {0, 1, 2}
#   sched_idx ∈ {0:none, 1:cosine}

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

# Unique experiment name so checkpoints don't collide between sweep tasks.
LR_TAG=$(echo "$LR" | sed 's/[.+]/p/g')
SCHED_TAG=$([ "$SCHEDULER" = "null" ] && echo "noscheduler" || echo "$SCHEDULER")
EXPERIMENT="stardist_sweep_${OPTIMIZER}_lr${LR_TAG}_${SCHED_TAG}"

echo "─────────────────────────────────────────────────"
echo "  optimizer = $OPTIMIZER"
echo "  lr        = $LR"
echo "  scheduler = $SCHEDULER"
echo "  experiment_name = $EXPERIMENT"
echo "─────────────────────────────────────────────────"

# parameters.scheduler=null tells Hydra to set the field to Python None;
# the pipeline's setup_scheduler(None) then skips wiring a scheduler.
# scheduler_kwargs.t_max=100 only kicks in when scheduler=cosine.
srun --unbuffered python lightning-stardist.py \
    train_data_paths=xenopus_jeanzay \
    train_data_paths.experiment_name="$EXPERIMENT" \
    parameters.optimizer="$OPTIMIZER" \
    parameters.learning_rate="$LR" \
    parameters.scheduler="$SCHEDULER" \
    parameters.scheduler_kwargs.t_max=100 \
    parameters.scheduler_kwargs.eta_min=1.0e-6
