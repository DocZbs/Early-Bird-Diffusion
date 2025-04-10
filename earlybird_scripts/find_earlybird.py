# Given a folder of pruned masks, find the EB-Ticket.
# This is accomplished by tracking Hamming Distances between masks.

import numpy as np
import os

dataset = "cifar10"
method = "magnitude"
prune_rate = "0.3"
lr = ""#"biglr-" #""#"biglr-"#"" #"biglr-", "midlr-"

end_epoch=99 # epoch to end the tracking
start_epoch=0 # epoch to start the tracking
epochs = end_epoch - start_epoch + 1
overlap = np.zeros((epochs, epochs))
masks = []
masks_list = []

# convergence threshold
eta = 0.1
# fifo length
fifo_length=5

for i in range(start_epoch, end_epoch+1):
    early_bird=True
    mask = np.load(f"checkpoints/pruned/cifar10/masks/{dataset}_pruned-{lr}{prune_rate}-{method}-{str(i)}.npy")
    masks.append(mask)
    if len(masks_list) < fifo_length:
        masks_list.append(mask)
    else:
        masks_list.pop(0)
        masks_list.append(mask)
    dists = []
    if len(masks_list) == fifo_length:
        
        for j in range(len(masks_list)-1):
            mask_i = masks_list[-1]
            mask_j = masks_list[j]
            dists.append(1 - float(np.sum(np.equal(mask_i , mask_j)!=0)) / mask_j.shape[0])
    if len(dists)==fifo_length-1:
        print(dists)
        for j in range(len(dists)):
            if dists[j] > 0.1:
                early_bird =False
                break
    if early_bird==True and len(dists) == fifo_length-1:
        print("EarlyBird found at epoch: ", i)
        break