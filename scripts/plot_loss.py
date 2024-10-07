import glob
from torch import load
import numpy as np
import matplotlib.pyplot as plt

# Parameters
model_dir = '/autofs/space/panamint_001/users/iglesias/models_temp/ResSR_test_fMRI'

g = sorted(glob.glob(model_dir + '/*.pth'))
loss = np.zeros(len(g))
for i in range(len(g)):
    loss[i] = load(g[i])['loss']
    print(loss[i])

# print(loss)
plt.figure(0)
plt.plot(loss)
plt.show()
