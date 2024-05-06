#!/bin/bash -x
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32gb
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=2
#SBATCH --wait-all-nodes=1
#SBATCH --job-name=attention
#SBATCH --output=slurm/%x_%j.out
#SBATCH --time=02:00:00
#SBATCH --qos scavenger

echo "Activating conda environment..."
eval "$(/nas/ucb/tutrinh/anaconda3/bin/conda shell.bash hook)"
echo "Activating conda environment.........."
conda activate chai
echo "Activated!"

export CUDA_LAUNCH_BLOCKING=1
export MASTER_PORT=8080
echo "Set CUDA environment!"

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "Set master addr to $MASTER_ADDR!"

cd /nas/ucb/tutrinh/backup/NoisyCaptioning
export PYTHONPATH="$PYTHONPATH:$PWD/src"
echo "Set directory!"

echo "Starting eval......"
python3 src/training/main.py \
	--val-data="/nas/ucb/tutrinh/cc3m/cc3m_val/{00000..00001}.tar" \
	--dataset-type webdataset \
	--model "coca_ViT-B-32" \
	--name "eval_testing_2" \
	--with-attention-scores \
	--eval \
	--eval-log-dir="/nas/ucb/tutrinh/backup/NoisyCaptioning/analysis/eval_logs/elr" \
	--eval-samples 64 \
	--eval-checkpoint-dir="/nas/ucb/tutrinh/backup/NoisyCaptioning/analysis/checkpoints/elr" \
	--eval-checkpoint-start 100 \
	--eval-checkpoint-end 100 \
	--eval-checkpoint-interval 10

