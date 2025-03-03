# Example command
# NUMBER_OF_EPOCHS=10
# for pruner in magnitude ours taylor random
#     do
#     for (( t=0; t<=NUMBER_OF_EPOCHS; t++ ))
#         do
#         for pruneratio in 0.3 #0.5 0.7

#             do
#             python prune.py \
#             --config cifar10.yml \
#             --exp checkpoints \
#             --timesteps 100 \
#             --eta 0 \
#             --ni \
#             --doc cifar10 \
#             --pruning_ratio $pruneratio  \
#             --use_ema \
#             --thr 0.05 \
#             --restore_from checkpoints/logs/cifar10/ckpt_ep$t.pth \
#             --pruner $pruner \
#             --save_pruned_model cifar10_pruned-$pruneratio-$pruner-$t.pth \
#             --save_masks

#             done
#         done
#     done