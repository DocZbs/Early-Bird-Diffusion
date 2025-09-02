torchrun \
    --nproc_per_node=2 \
    main.py \
    --config cifar10.yml \
    --exp checkpoints \
    --doc cifar10 \
    --ni \
    --use_ema \
    --train_from_scratch \
    --skip_type quad

python  main.py \
 --config cifar10.yml \
 --exp checkpoints \
 --doc cifar10 \
 --ni \
 --use_ema \
 --train_from_scratch \
 --skip_type quad  

# cifar10 with big learning rate
python main.py \
    --config cifar10_bne-2.yml \
    --exp checkpoints \
    --doc cifar10_test_biglr \
    --ni \
    --use_ema \
    --train_from_scratch

# cifar10 with middle learning rate
python main.py \
    --config cifar10_bne-3.yml \
    --exp checkpoints \
    --doc cifar10_test_midlr \
    --ni \
    --use_ema \
    --train_from_scratch

# cifar 10 with bn
python main.py \
    --config cifar10_bn.yml \
    --exp checkpoints \
    --doc cifar10_bn \
    --ni \
    --use_ema \
    --train_from_scratch \
    --sr



# cifar 10 with bn
python main.py \
    --config cifar10_pruning.yml \
    --exp checkpoints \
    --doc cifar10_pruning \
    --ni \
    --use_ema \
    --train_from_scratch \
    --sr
# training  on LSUN
python main.py \
    --config church.yml \
    --exp checkpoints \
    --doc church_test \
    --ni \
    --use_ema \
    --train_from_scratch

