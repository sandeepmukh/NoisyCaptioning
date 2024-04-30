export CUDA_VISIBLE_DEVICES=3,4,7
export OMP_NUM_THREADS=12
export CUDA_LAUNCH_BLOCKING=1
torchrun --nproc_per_node 4 --master_port 8080 -m training.main \
	--train-data="/nas/ucb/tutrinh/cc3m/cc3m/{00000..00331}.tar" \
	--train-num-samples 1000000 \
	--val-data="/nas/ucb/tutrinh/cc3m/cc3m_val/{00000..00001}.tar" \
	--val-num-samples 10000 \
	--imagenet-val "/nas/ucb/tutrinh/imagenet/validation" \
	--dataset-type webdataset \
	--batch-size 128 \
	--warmup 2000 \
	--epochs 100 \
	--lr 5e-4 \
	--precision amp \
	--workers 4 \
	--elr-distill \
	--model "coca_ViT-B-32" \
	--name "coca_cc3m_elr_1" \
	--report-to "wandb" \
	--wandb-project-name "open-clip-cc3m-elr" \
	--gather-with-grad \
	--local-loss \
	--save-frequency 5 \
	--torchcompile \

