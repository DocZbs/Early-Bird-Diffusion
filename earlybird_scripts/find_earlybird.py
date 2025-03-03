import numpy as np
import os
dataset = "cifar10"
method = "magnitude"
prune_rate = "0.3"
lr = ""#"biglr-" #""#"biglr-"#"" #"biglr-", "midlr-"
# for i in range(500):
#     mask =
#     print(mask.shape)

end_epoch=99
start_epoch=0
epochs = end_epoch - start_epoch + 1
overlap = np.zeros((epochs, epochs))
# save_dir = os.path.join(args.save, 'overlap_'+str(args.percent))
masks = []
masks_list = []

for i in range(start_epoch, end_epoch+1):
    early_bird=True
    # resume = args.save + 'model_' + str(i-1) + '.pth'
    # checkpoint = torch.load(resume)
    # model.load_state_dict(checkpoint)
    mask = np.load(f"checkpoints/pruned/cifar10/masks/{dataset}_pruned-{lr}{prune_rate}-{method}-{str(i)}.npy")
    masks.append(mask)
    if len(masks_list) < 5:
        masks_list.append(mask)
    else:
        masks_list.pop(0)
        masks_list.append(mask)
    dists = []
    if len(masks_list) == 5:
        
        for j in range(len(masks_list)-1):
            mask_i = masks_list[-1]
            mask_j = masks_list[j]
            dists.append(1 - float(np.sum(np.equal(mask_i , mask_j)!=0)) / mask_j.shape[0])
    if len(dists)==4:
        print(dists)
        for j in range(len(dists)):
            if dists[j] > 0.1:
                early_bird =False
                break
    if early_bird==True and len(dists) == 4:
        print("EarlyBird found at epoch:", i)
        break