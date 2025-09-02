# DIT version sampling commands

# Sample from DIT trained model with FID evaluation
torchrun \
    --nproc_per_node=2 \
    --master_port=29501 \
    main_dit.py \
    --config cifar10_dit_vrdit.yml \
    --exp checkpoints \
    --doc cifar10_vrdit \
    --ni \
    --sample \
    --fid \
    --skip_type quad \
    --timesteps 100 \
    --image_folder samples/cifar10_vrdit_epo200 \
    --restore_from checkpoints/logs/cifar10_vrdit/ckpt_ep200.pth


# Sample from VR-DiT trained model
torchrun \
    --nproc_per_node=2 \
    --master_port=29501 \
    main_dit.py \
    --config cifar10_dit_vrdit.yml \
    --exp checkpoints \
    --doc cifar10_dit_vrdit_sample \
    --ni \
    --sample \
    --fid \
    --skip_type uniform \
    --timesteps 250 \
    --image_folder samples/cifar10_dit_vrdit \
    --restore_from checkpoints/logs/cifar10_dit_vrdit/ckpt_20000.pth

# Sample with sequence generation (denoising process visualization)
python main_dit.py \
    --config cifar10_dit.yml \
    --exp checkpoints \
    --doc cifar10_dit_sequence \
    --ni \
    --sample \
    --sequence \
    --skip_type quad \
    --timesteps 100 \
    --image_folder samples/cifar10_dit_sequence \
    --restore_from checkpoints/logs/cifar10_dit/ckpt.pth

# Sample with interpolation
python main_dit.py \
    --config cifar10_dit.yml \
    --exp checkpoints \
    --doc cifar10_dit_interp \
    --ni \
    --sample \
    --interpolation \
    --skip_type quad \
    --timesteps 100 \
    --image_folder samples/cifar10_dit_interpolation \
    --restore_from checkpoints/logs/cifar10_dit/ckpt.pth

# Sample from specific checkpoint step
python main_dit.py \
    --config cifar10_dit.yml \
    --exp checkpoints \
    --doc cifar10_dit_step \
    --ni \
    --sample \
    --fid \
    --skip_type quad \
    --timesteps 100 \
    --image_folder samples/cifar10_dit_step50k \
    --restore_from checkpoints/logs/cifar10_dit/ckpt_50000.pth

# Sample with DDPM sampling method
python main_dit.py \
    --config cifar10_dit.yml \
    --exp checkpoints \
    --doc cifar10_dit_ddpm \
    --ni \
    --sample \
    --fid \
    --sample_type ddpm_noisy \
    --skip_type quad \
    --timesteps 100 \
    --image_folder samples/cifar10_dit_ddpm \
    --restore_from checkpoints/logs/cifar10_dit/ckpt.pth

# Sample with different timestep settings
python main_dit.py \
    --config cifar10_dit.yml \
    --exp checkpoints \
    --doc cifar10_dit_fast \
    --ni \
    --sample \
    --fid \
    --skip_type quad \
    --timesteps 50 \
    --image_folder samples/cifar10_dit_fast \
    --restore_from checkpoints/logs/cifar10_dit/ckpt.pth