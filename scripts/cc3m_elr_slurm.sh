#!/bin/bash -x
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=6
#SBATCH --wait-all-nodes=1
#SBATCH --job-name=cc3m_elr
#SBATCH --output=slurm/%j.out
#SBATCH --time=72:00:00
#SBATCH --qos scavenger

echo "Activating conda environment..."
eval "$(/nas/ucb/tutrinh/anaconda3/bin/conda shell.bash hook)"
echo "Activating conda environment.........."
conda activate chai
echo "Activated!"

echo "Logging into wandb..."
wandb login 4a6017fc91542ffdb82ee3d6213e9cf0c11fd892
echo "Logged into wandb!"

export CUDA_LAUNCH_BLOCKING=1
export MASTER_PORT=8080
echo "Set CUDA environment!"

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "Set master addr to $MASTER_ADDR!"

cd /nas/ucb/tutrinh/NoisyCaptioning
export PYTHONPATH="$PYTHONPATH:$PWD/src"
echo "Set directory!"

echo "STARTING TRAINING........"
srun --cpu_bind=v --accel-bind=gn python -u src/training/main.py \
	--save-frequency 5 \
	--train-data="/nas/ucb/tutrinh/cc3m/cc3m/{00000..00331}.tar" \
	--train-num-samples 1000000 \
	--val-data="/nas/ucb/tutrinh/cc3m/cc3m_val/{00000..00001}.tar" \
	--val-num-samples 10000 \
	--imagenet-val "/nas/ucb/tutrinh/imagenet/validation" \
	--dataset-type webdataset \
	--batch-size 128 \
	--warmup 2000 \
	--epochs 100 \
	--workers 8 \
	--lr 5e-4 \
	--precision amp \
	--elr-distill \
	--model "coca_ViT-B-32" \
	--name "coca_cc3m_elr_4" \
	--report-to "wandb" \
	--wandb-project-name "open-clip-cc3m-elr" \
	--gather-with-grad \
	--local-loss \
	--torchcompile
echo "TRAINING STARTED.........."

