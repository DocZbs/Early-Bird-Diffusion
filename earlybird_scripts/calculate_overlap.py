# Calculate the overlaps between epoch-wise masks.
# Used to plot heatmaps of EB tickets.

import numpy as np
import os

dataset="cifar10" # cifar10, celeba, etc
method = "magnitude" # magnitude, ours, taylor, random
prune_rate = "0.3"
lr = "" # "midlr-"#"" #"biglr-", "midlr-"
resume = "statistic"

end_epoch=10
start_epoch=0
epochs = end_epoch - start_epoch + 1
overlap = np.zeros((epochs, epochs))
masks = []

for i in range(start_epoch, end_epoch+1):
    
    mask = np.load(f"./checkpoints/pruned/cifar10/masks/{dataset}_pruned-{lr}{prune_rate}-{method}-{str(i)}.npy")
    masks.append(mask)

size = masks[0].shape[0]
for i in range(start_epoch, end_epoch+1):
    for j in range(start_epoch, end_epoch+1):
        overlap[i-1, j-1] = float(np.sum(np.equal(masks[i-1] , masks[j-1])!=0)) / size
        print('overlap[{}, {}] = {}'.format(i-1, j-1, overlap[i-1, j-1]))

# Save the overlap mask
# Used for plotting the heatmaps
np.save(os.path.join(resume, f"{dataset}-{lr}{method}-{prune_rate}-mask"), overlap)

