import numpy as np
import os
import matplotlib.pyplot as plt

dataset="cifar10"
method ="magnitude"
resume = 'statistic/' 
lr = ""#"biglr-"#"" #"biglr-", "midlr-"
path_30 = resume + f'{dataset}-{lr}{method}-0.3-mask.npy'
path_50 = resume + f'{dataset}-{lr}{method}-0.3-mask.npy'
path_70 = resume + f'{dataset}-{lr}{method}-0.3-mask.npy'

overlap_30 = np.load(path_30)
overlap_50 = np.load(path_50)
overlap_70 = np.load(path_70)

# print(overlap[100, :])
fig = plt.figure('overlap', figsize=(20,10))
ax1 = fig.add_subplot(1,3,1)
# for i in range(overlap_30.shape[0]):
#     for j in range(overlap_30.shape[1]):
#         if overlap_30[i][j]>=0.9:
#             overlap_30[i][j]=0.99
#         else:
#             overlap_30[i][j]=0.01
# overlap_30[np.where(overlap_30>=0.9)[0]]=1.0
overlap = ax1.imshow(overlap_30, cmap=plt.cm.viridis, interpolation='none', vmin=0, vmax=1)
cb = plt.colorbar(overlap, fraction=0.046, pad=0.04)
ax1.set_title('Correlation map (p=0.3)', fontsize=20)
ax1.set_xlabel('epochs', fontsize=15)
ax1.set_ylabel('epochs', fontsize=15)
ax2 = fig.add_subplot(1,3,2)
overlap = ax2.imshow(overlap_50, cmap=plt.cm.viridis, interpolation='none', vmin=0, vmax=1)
cb = plt.colorbar(overlap, fraction=0.046, pad=0.04)
ax2.set_title('Correlation map (p=0.5)', fontsize=20)
ax2.set_xlabel('epochs', fontsize=15)
ax2.set_ylabel('epochs', fontsize=15)
ax3 = fig.add_subplot(1,3,3)
overlap = ax3.imshow(overlap_70, cmap=plt.cm.viridis, interpolation='none', vmin=0, vmax=1)
cb = plt.colorbar(overlap, fraction=0.046, pad=0.04)
ax3.set_title('Correlation map (p=0.7)', fontsize=20)
ax3.set_xlabel('epochs', fontsize=15)
ax3.set_ylabel('epochs', fontsize=15)

plt.savefig(os.path.join(resume, f'{dataset}-{lr}{method}.png'))
