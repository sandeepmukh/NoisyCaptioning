export CUDA_VISIBLE_DEVICES=5
export OMP_NUM_THREADS=12
export CUDA_LAUNCH_BLOCKING=1

python3 -m training.main \
	--val-data="/nas/ucb/tutrinh/cc3m/cc3m_val/{00000..00001}.tar" \
	--dataset-type webdataset \
	--model "coca_ViT-B-32" \
	--name "eval_testing_1" \
	--eval \
	--eval-log-dir="/nas/ucb/tutrinh/backup/NoisyCaptioning/analysis/eval_logs/baseline" \
	--eval-samples 1 \
	--eval-save-samples-dir="/nas/ucb/tutrinh/backup/NoisyCaptioning/analysis/eval_samples" \
	--eval-checkpoint-dir="/nas/ucb/tutrinh/backup/NoisyCaptioning/analysis/checkpoints/baseline" \
	--eval-checkpoint-start 100 \
	--eval-checkpoint-end 100 \
	--eval-checkpoint-interval 10

