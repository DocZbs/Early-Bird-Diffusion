# DIT version training commands
torchrun \
    --nproc_per_node=2 \
    main_dit.py \
    --config cifar10_dit_vrdit.yml \
    --exp checkpoints \
    --doc cifar10_dit_vrdit \
    --ni \
    --use_ema \
    --train_from_scratch \
    --skip_type quad
# Multi-GPU training with torchrun (recommended)
torchrun \
    --nproc_per_node=2 \
    main_dit.py \
    --config cifar10_dit.yml \
    --exp checkpoints \
    --doc cifar10_dit \
    --ni \
    --use_ema \
    --train_from_scratch \
    --skip_type quad

# Single GPU training
python main_dit.py \
    --config cifar10_dit.yml \
    --exp checkpoints \
    --doc cifar10_dit \
    --ni \
    --use_ema \
    --train_from_scratch \
    --skip_type quad

# DIT with larger model (increased depth and hidden size)
python main_dit.py \
    --config cifar10_dit_large.yml \
    --exp checkpoints \
    --doc cifar10_dit_large \
    --ni \
    --use_ema \
    --train_from_scratch \
    --skip_type quad

# DIT with different patch size
python main_dit.py \
    --config cifar10_dit_patch4.yml \
    --exp checkpoints \
    --doc cifar10_dit_patch4 \
    --ni \
    --use_ema \
    --train_from_scratch \
    --skip_type quad

# VR-DiT training with variance reduction techniques
torchrun \
    --nproc_per_node=2 \
    main_dit.py \
    --config cifar10_dit_vrdit.yml \
    --exp checkpoints \
    --doc cifar10_dit_vrdit \
    --ni \
    --use_ema \
    --train_from_scratch \
    --skip_type uniform