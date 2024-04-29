BASE_PATH=/home/sandeepmukh/open_clip

export CUDA_VISIBLE_DEVICES=0
clip_benchmark eval --dataset=mscoco_captions --task=captioning \
  --pretrained=$BASE_PATH/src/logs/coca_coyo_baseline_4/checkpoints/epoch_100.pt \
  --model=coca_ViT-B-32 --output=$BASE_PATH/src/logs/coca_coyo_baseline_4/coco_results_captioning.json --batch_size=64 \
  --split val --dataset_root $BASE_PATH/data/mscoco &

sleep 1

export CUDA_VISIBLE_DEVICES=1
clip_benchmark eval --dataset=mscoco_captions --task=zeroshot_retrieval \
  --pretrained=$BASE_PATH/src/logs/coca_coyo_baseline_4/checkpoints/epoch_100.pt \
  --model=coca_ViT-B-32 --output=$BASE_PATH/src/logs/coca_coyo_baseline_4/coco_results_retrieval.json --batch_size=64 \
  --split val --dataset_root $BASE_PATH/data/mscoco &

sleep 1

export CUDA_VISIBLE_DEVICES=2
clip_benchmark eval --dataset=mscoco_captions --task=captioning \
  --pretrained=$BASE_PATH/src/logs/coca_coyo_elr_11/checkpoints/epoch_100.pt \
  --model=coca_ViT-B-32 --output=$BASE_PATH/src/logs/coca_coyo_elr_11/coco_results_captioning.json --batch_size=64 \
  --split val --dataset_root $BASE_PATH/data/mscoco &

sleep 1

export CUDA_VISIBLE_DEVICES=3
clip_benchmark eval --dataset=mscoco_captions --task=zeroshot_retrieval \
  --pretrained=$BASE_PATH/src/logs/coca_coyo_elr_11/checkpoints/epoch_100.pt \
  --model=coca_ViT-B-32 --output=$BASE_PATH/src/logs/coca_coyo_elr_11/coco_results_retrieval.json --batch_size=64 \
  --split val --dataset_root $BASE_PATH/data/mscoco 