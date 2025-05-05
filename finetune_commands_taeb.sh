#!/bin/bash

# Change to your working directory
cd /storage/scratch1/1/lwhalen7/deb

# Activate the conda environment
source /storage/home/hcoda1/1/lwhalen7/anaconda3/etc/profile.d/conda.sh
conda activate earlybird

# export CUDA_VISIBLE_DEVICES=0
# python main_birds.py \
#  --config celeba.yml \
#  --timesteps 100 \
#  --eta 0 \
#  --ni \
#  --exp checkpoints \
#  --doc timestepbird_celeba_30\
#  --skip_type quad  \
#  --use_ema \
#  --bird_pruned_model_path checkpoints/pruned/celeba_find_eb_03/models/celeba_pruned-0.3-magnitude-8.pth\
#  --train_from_scratch \
#  --ts_lower_bound 0 \
#  --ts_upper_bound 260

# export CUDA_VISIBLE_DEVICES=0
# python main_birds.py \
#  --config cifar10.yml \
#  --timesteps 100 \
#  --eta 0 \
#  --ni \
#  --exp checkpoints \
#  --doc a6000_44_pruned\
#  --skip_type quad  \
#  --use_ema \
#  --bird_pruned_model_path checkpoints/pruned/cifar10_pruned_eb/models/cifar10-0.44-magnitude-0.pth\
#  --train_from_scratch \
#  --ts_lower_bound 0\
#  --ts_upper_bound 460

# export CUDA_VISIBLE_DEVICES=0
# python main_birds.py \
#  --config cifar10.yml \
#  --timesteps 100 \
#  --eta 0 \
#  --ni \
#  --exp checkpoints \
#  --doc timestepbird_cifar10_80_440_720\
#  --skip_type quad  \
#  --use_ema \
#  --bird_pruned_model_path /storage/home/hcoda1/1/lwhalen7/scratch/deb/checkpoints/pruned/cifar10_eb_08/models/cifar10_eb_08-0.8-magnitude-7.pth\
#  --train_from_scratch \
#  --ts_lower_bound 440\
#  --ts_upper_bound 720

# export CUDA_VISIBLE_DEVICES=0
# python main_birds.py \
#  --config cifar10.yml \
#  --timesteps 100 \
#  --eta 0 \
#  --ni \
#  --exp checkpoints \
#  --doc timestepbird_cifar10_80_440_720\
#  --skip_type quad  \
#  --use_ema \
#  --bird_pruned_model_path /storage/home/hcoda1/1/lwhalen7/scratch/deb/checkpoints/pruned/cifar10_eb_08/models/cifar10_eb_08-0.8-magnitude-7.pth\
#  --train_from_scratch \
#  --ts_lower_bound 440\
#  --ts_upper_bound 740

# export CUDA_VISIBLE_DEVICES=0
# python main_birds.py \
#  --config cifar10.yml \
#  --timesteps 100 \
#  --eta 0 \
#  --ni \
#  --exp checkpoints \
#  --doc timestepbird_cifar10_80_720_1000\
#  --skip_type quad  \
#  --use_ema \
#  --bird_pruned_model_path /storage/home/hcoda1/1/lwhalen7/scratch/deb/checkpoints/pruned/cifar10_eb_08/models/cifar10_eb_08-0.8-magnitude-7.pth\
#  --train_from_scratch \
#  --ts_lower_bound 720\
#  --ts_upper_bound 1000

# export CUDA_VISIBLE_DEVICES=0
# python main_birds.py \
#  --config celeba.yml \
#  --timesteps 100 \
#  --eta 0 \
#  --ni \
#  --exp checkpoints \
#  --doc timestepbird_celeba_80\
#  --skip_type quad  \
#  --use_ema \
#  --bird_pruned_model_path checkpoints/pruned/celeba_find_eb_08/models/celeba_pruned-0.8-magnitude-6.pth\
#  --train_from_scratch \
#  --ts_lower_bound 440 \
#  --ts_upper_bound 1000

# export CUDA_VISIBLE_DEVICES=0
# python main_birds.py \
#  --config cifar10.yml \
#  --timesteps 100 \
#  --eta 0 \
#  --ni \
#  --exp checkpoints \
#  --doc timestep_eb_60_ts240_460 \
#  --skip_type quad  \
#  --use_ema \
#  --bird_pruned_model_path checkpoints/pruned/cifar10_eb_06/models/cifar10_eb_06-0.6-magnitude-12.pth\
#  --train_from_scratch \
#  --ts_lower_bound 240 \
#  --ts_upper_bound 460

# export CUDA_VISIBLE_DEVICES=0
# python main_birds.py \
#  --config cifar10.yml \
#  --timesteps 100 \
#  --eta 0 \
#  --ni \
#  --exp checkpoints \
#  --doc timestep_eb_80_ts440_1000 \
#  --skip_type quad  \
#  --use_ema \
#  --bird_pruned_model_path checkpoints/pruned/cifar10_eb_08/models/cifar10_eb_08-0.8-magnitude-7.pth\
#  --train_from_scratch \
#  --ts_lower_bound  440\
#  --ts_upper_bound 1000

export CUDA_VISIBLE_DEVICES=0
python main_birds.py \
 --config cifar10.yml \
 --timesteps 100 \
 --eta 0 \
 --ni \
 --exp checkpoints \
 --doc a6000_0.3_cifar10\
 --skip_type quad  \
 --use_ema \
 --bird_pruned_model_path /storage/home/hcoda1/1/lwhalen7/scratch/deb/checkpoints/pruned/cifar10_eb_03/models/cifar10_eb_03-0.3-magnitude-10.pth\
 --train_from_scratch \
 --ts_lower_bound 0\
 --ts_upper_bound 260