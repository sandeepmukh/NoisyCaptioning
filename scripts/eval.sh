export CUDA_VISIBLE_DEVICES=1,3
export OMP_NUM_THREADS=12
export CUDA_LAUNCH_BLOCKING=1

python -m training.main \
	--val-data="/nas/ucb/tutrinh/cc3m/cc3m_val/{00000..00001}.tar" \
	--dataset-type webdataset \
	--model "coca_ViT-B-32" \
	--name "eval_testing_1" \
	--eval \
	--eval-log-dir="/nas/ucb/tutrinh/backup/NoisyCaptioning/analysis/eval_logs/baseline" \
	--eval-samples 1 \
	--eval-checkpoint-dir="/nas/ucb/tutrinh/backup/NoisyCaptioning/analysis/checkpoints/baseline" \
	--eval-checkpoint-start 5 \
	--eval-checkpoint-end 5 \
	--eval-checkpoint-interval 10
