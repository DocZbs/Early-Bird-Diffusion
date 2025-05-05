# Early-Bird Diffusion: Investigating and Leveraging Timestep-Aware Early-Bird Tickets in Diffusion Models for Efficient Training
Lexington Whalen*, Zhenbang Du*, Haoran You*, Chaojian Li, Sixu Li, and Yingyan (Celine) Lin.
*Equal Contribution

Accepted by [**CVPR 2025**](https://cvpr.thecvf.com/).

## Overview
### Motivation and Insight:
Training diffusion models (DMs) is computationally expensive due to multiple forward and backward passes across numerous timesteps. We investigate the existence of "Early-Bird (EB) tickets" - sparse subnetworks that emerge early in training yet maintain high generation quality - and develop a new approach called EB-Diff-Train to leverage these tickets for efficient training.

### Implementation:
We identify both traditional EB tickets (consistent across all timesteps) and introduce timestep-aware EB (TA-EB) tickets that apply varying sparsity levels according to timestep importance. Our EB-Diff-Train method trains these region-specific tickets in parallel and combines them during inference, optimizing training both spatially (through model pruning) and temporally (through parallelism).

### Results:
The proposed EB-Diff-Train method achieves 2.9×-5.8× training speedups over unpruned dense models and up to 10.3× faster training compared to standard train-prune-finetune approaches without compromising generation quality. Our method is orthogonal to and can be combined with other diffusion training acceleration techniques such as [SpeeD](https://github.com/NUS-HPC-AI-Lab/SpeeD), further enhancing performance.


## Code Usage
### Environment Setup
```
conda env create -n earlybird python=3.9
conda activate earlybird
pip install -r requirements.txt
```

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

## Citation

If you find this work useful for your research, please cite:


````
@inproceedings{whalen2025earlybird,
  title={Early-Bird Diffusion: Investigating and Leveraging Timestep-Aware Early-Bird Tickets in Diffusion Models for Efficient Training},
  author={Lexington Whalen, Zhenbang Du, Haoran You, Chaojian Li, Sixu Li, and Yingyan (Celine) Lin.},
  booktitle={The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2025)},
  year={2025}
}
````