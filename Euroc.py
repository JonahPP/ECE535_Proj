import numpy as np
import os 
import pandas as pd 
import matplotlib.pyplot as plt


"""
for easy change:
MH_05
"""
gt = pd.read_csv("Euroc_gt\MH_05.csv", header=None)
est = np.loadtxt("Euroc\\MH_05.txt")
# getting rid of index 0 until 8 to preserve 1-7 as xyz-quad
# we modulo 10 so that it matches the results, as it has close to x10 amt of data
gt_spliced = gt.iloc[1:, 1:8]
gt_scaled = gt_spliced[::10]
gt_official = gt_scaled.iloc[:-1]

# at this point, both gt and est are 3638 x 7, meaning it's ready
# for tartanVO  print(gt_official.shape, est.shape)
# change the pd dataframe to numpmy ndarray for easy plotting
gt_official = np.array(gt_official)


# only need the first 3 for x, y
cut_est = est[:, :3]
trans_est = cut_est.T
cut_gt = gt_official[:, :3]
trans_gt = cut_gt.T

trans_gt = trans_gt.astype(np.float64)


fig = plt.figure(figsize=(7,6))
ax= fig.add_subplot(111)

plt.plot(trans_est[0], trans_est[1], label='Estimated', color='orange')
plt.plot(trans_gt[0], trans_gt[1], label='Ground Truth', color='black', linestyle='dashed')

ax.set_title('TartanVO Graph for MH_05')
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.legend()
plt.show()


