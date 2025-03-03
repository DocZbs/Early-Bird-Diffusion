# Example commands
# sample from official pretrained models
# python main.py \
#  --config cifar10.yml \
#  --exp checkpoints \
#  --doc cifar10_test \
#  --ni \
#  --sample \
#  --fid \
#  --use_ema   \
#  --skip_type quad \
#  --timesteps 100 \
#  --image_folder ori  

# # sample from our trained models
# python main.py \
#  --config cifar10.yml \
#  --exp checkpoints \
#  --doc cifar10_test \
#  --ni \
#  --sample \
#  --fid \
#  --use_ema   \
#  --skip_type quad \
#  --timesteps 100 \
#  --image_folder ours_init  \
#  --restore_from checkpoints/logs/cifar10/ckpt_init.pth 
 
# # sample from our (fine-tuned) pruned models
# python  main.py \
#  --config cifar10.yml \
#  --exp checkpoints \
#  --doc cifar10_pruned_test \
#  --ni \
#  --sample \
#  --fid \
#  --use_ema  \
#  --skip_type quad \
#  --timesteps 100 \
#  --load_pruned_model   checkpoints/logs/cifar10_eb_magnitude_0.3/ckpt_1.pth \
#  --image_folder magnitude_eb_0.3_step1  \
#  --base_pruned_model  checkpoints/pruned/cifar10/models/cifar10_pruned-0.3-magnitude-4.pth 

