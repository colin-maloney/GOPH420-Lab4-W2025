import numpy as np

data = np.loadtxt('../data/M_data1.txt')
t = data[:,0]

t_max1 = np.argwhere(t<=35)[-1]





