import numpy as np 

from src.goph_lab04.regression import multiregression 

def main(): 
    data = np.leatxt(".../data/M_data1.txt") 

    t = data[:, 0] 
    m_data = data[:, 1] 

    int_1 = np.argwhere(t < 35)[-1] 

    m = np.linespace(-0.5, 1.5, 19) 
    n = np.zeros_like(m) 

    for i, mm in enumerate(m): 
        n[i] = np.sum(m_data[int_1:] > mm)

    y = np.log(n) 
    z = np.vstack((np.ones_like(m), m)).T 

    aCoeff, em, R2 = multiregression(y, z) 

    print(f"aCoeff: {aCoeff}")
    print(f"R^2: {R2}") 
    print(f" residuals: {em}")
