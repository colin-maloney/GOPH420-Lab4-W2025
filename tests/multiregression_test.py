import numpy as np
from goph_lab04.regression import multiregression

y_data = np.loadtxt('../tests/test_data.txt')
y = y_data[:,0]

Z_data = np.loadtxt('../tests/test_data.txt')
Z = Z_data[:,1:]

coefficients = multiregression(y, Z)[0]
error = multiregression(y, Z)[1]
R2 = multiregression(y, Z)[2]

print(f'coefficients: {coefficients}')
print(f'error: {error}')
print(f'R2: {R2}')
