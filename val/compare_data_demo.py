import numpy as np

data = np.load("y.npy")
data2 = np.load("y2.npy")

data = np.squeeze(data)
data2 = np.squeeze(data2)

different = data-data2

print(np.sum(different))