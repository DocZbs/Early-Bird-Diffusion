#!/bin/bash

# UNet with Timestep Importance Sampling Training
torchrun \
    --nproc_per_node=2 \
    main.py \
    --config cifar10_unet_importance.yml \
    --exp checkpoints \
    --doc unet_importance \
    --ni \
    --use_ema \
    --train_from_scratch \
    --skip_type quad

# Alternative: Single GPU training
# python main.py \
#     --config cifar10_unet_importance.yml \
#     --exp checkpoints \
#     --doc unet_importance_single \
#     --ni \
#     --use_ema \
#     --train_from_scratch \
#     --skip_type quad \
#     --single_gpu