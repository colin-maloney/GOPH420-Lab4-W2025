import numpy as np
import matplotlib.pyplot as plt

from src.goph_lab04.regression import multiregression 

def main(): 
    data = np.loadtxt("../data/M_data1.txt")

    t = data[:, 0] 
    m_data = data[:, 1] 

    int_1 = np.argwhere(t < 35)[-1].item()

    m = np.linspace(-0.2, 1.0, 20)
    n = np.zeros_like(m)

    for i, mm in enumerate(m): 
        n[i] = np.count_nonzero(m_data[:int_1] > mm)

    y = np.log(n) 
    z = np.vstack((np.ones_like(m), m)).T 

    aCoeff, em, R2 = multiregression(y, z) 

    print(f"aCoeff: {aCoeff}")
    print(f"R^2: {R2}") 
    print(f" residuals: {em}")

    y_model = []
    for i, mag in enumerate(m_data[:int_1]):
        y_model.append(aCoeff[0] + aCoeff[1] * mag)

    plt.plot(m_data[:int_1], y_model)
    plt.show()
if __name__ == "__main__":
    main()

