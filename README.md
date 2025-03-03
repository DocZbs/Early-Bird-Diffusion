# Early-Bird Diffusion: Investigating and Leveraging Timestep-Aware Early-Bird Tickets in Diffusion Models for Efficient Training
Lexington Whalen, Zhenbang Du, Haoran You, Chaojian Li, Sixu Li, and Yingyan (Celine) Lin.

Accepted as a CVPR 2025 Paper.

## Overview
### Motivation and Insight: 
TODO
### Implementation: 
TODO
### Results: 
TODO

## Code Usage
### Environment Setup
- conda env create -n earlybird python=3.9
- conda activate earlybird
- pip install -r requirements.txt

### Workflow
Train the model from scratch using commands in train_commands.sh:
```
python main.py \
 --config cifar10.yml \
 --exp checkpoints \
 --doc cifar10 \
 --ni \
 --use_ema \
 --train_from_scratch \
 --skip_type quad
```

This command saves per-epoch trained checkpoints in checkpoints/logs/cifar10.
Prune the checkpoints using commands in `prune_commands.sh`.

Find the early-bird epoch using `earlybird_scripts/find_earlybird.py`

Fine-tune the pruned model using commands in `finetune_commands.sh`.

Generate images using commands in `sample_commands.sh`.

Calculate FID score using `cal_fid.sh`.


This codebase is inspired by NeurIPS 2023's [Structural Pruning for Diffusion Models](https://arxiv.org/pdf/2305.10924).