export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=12
torchrun --nproc_per_node 4 --master_port 8888 -m training.main \
         --train-data="/home/sandeepmukh/imagen-pytorch/data/coyo-700m-webdataset/{00000..00340}.tar" \
         --train-num-samples 1000000 \
         --val-data="/home/sandeepmukh/imagen-pytorch/data/coyo-700m-webdataset/{00340..00345}.tar" \
         --val-num-samples 10000 \
         --dataset-type webdataset \
         --batch-size 128 \
         --warmup 2000 \
         --epochs 100 \
         --lr 5e-4 \
         --precision amp \
         --workers 3 \
         --model "coca_ViT-B-32" \
         --name "coca_coyo_elr_0" \
         --report-to "wandb" \
         --wandb-project-name "open-clip-baseline" \
         --imagenet-val "/home/sandeepmukh/open_clip/imagenet/validation" \
         --gather-with-grad \
         --local-loss \
         --save-frequency 5 \