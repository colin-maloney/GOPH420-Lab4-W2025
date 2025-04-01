import numpy as np
import matplotlib.pyplot as plt

from src.goph_lab04.regression import multiregression 

def main(): 
    data = np.loadtxt("../data/M_data1.txt")

    t = data[:, 0] 
    m_data = data[:, 1] 

    int_1 = np.argwhere(t < 35)[-1].item()

    m1 = np.linspace(-0.15, 0.8, 20)
    n1 = np.zeros_like(m1)

    for i, m1m in enumerate(m1): 
        n1[i] = np.count_nonzero(m_data[:int_1] > m1m)

    y1 = np.log10(n1)
    z1 = np.vstack((np.ones_like(m1), m1)).T 

    aCoeff1, em1, R2_1 = multiregression(y1, z1) 

    print(f"aCoeff: {aCoeff1}")
    print(f"R^2: {R2_1}") 
    print(f" residuals: {em1}")

    y_model1 = z1 @ aCoeff1

    plt.figure()
    plt.plot(m1, y_model1, "--k")
    plt.plot(m1,y1, "ok")
    plt.savefig("../figures/M_data1.png")

    int_2 = np.argwhere(t < 45)[-1].item() 

    m2 = np.linspace(-0.15, 0.8, 20)
    n2 = np.zeros_like(m2) 

    for i, m2m in enumerate(m2):
        n2[i] = np.count_nonzero(m_data[int_1:int_2] > m2m)

    y2 = np.log(n2) 
    z2 = np.vstack((np.ones_like(m2), m2)).T

    aCoeff2, em2, R2_2 = multiregression(y2, z2)

    print(f"aCoeff: {aCoeff2}") 
    print(f"R^2: {R2_2}") 
    print(f" residuals: {em2}")

    y_model2 = z2 @ aCoeff2

    int_3 = np.argwhere(t < 73)[-1].item()

    m3 = np.linspace(-0.15, 0.8, 20)
    n3 = np.zeros_like(m3) 

    for i, m3m in enumerate(m3):
        n3[i] = np.count_nonzero(m_data[int_2:int_3] > m3m)

    y3 = np.log(n3)
    z3 = np.vstack((np.ones_like(m3), m3)).T 

    aCoeff3, em3, R2_3 = multiregression(y3, z3) 
    print(f"aCoeff: {aCoeff3}")
    print(f"R^2: {R2_3}") 
    print(f" residuals: {em3}")

    y_model3 = z3 @ aCoeff3

    int_4 = np.argwhere(t < 96)[-1].item()

    m4 = np.linspace(-0.15, 0.8, 20)
    n4 = np.zeros_like(m4) 

    for i, m4m in enumerate(m4):
        n4[i] = np.count_nonzero(m_data[int_3:int_4] > m4m)

    y4 = np.log(n4) 
    z4 = np.vstack((np.ones_like(m4), m4)).T 

    aCoeff4, em4, R2_4 = multiregression(y4, z4) 
    print(f"aCoeff: {aCoeff4}")
    print(f"R^2: {R2_4}") 
    print(f" residuals: {em4}")

    y_model4 = z4 @ aCoeff4

    int_5 = np.argwhere(t < 120)[-1].item()

    m5 = np.linspace(-0.15, 0.8, 20)
    n = np.zeros_like(m5) 

    for i, m5m in enumerate(m5):
        n[i] = np.count_nonzero(m_data[int_4:int_5] > m5m)

    y5 = np.log(n) 
    z5 = np.vstack((np.ones_like(m5), m5)).T 

    aCoeff5, em5, R2_5 = multiregression(y5, z5)
    print(f"aCoeff: {aCoeff5}") 
    print(f"R^2: {R2_5}") 
    print(f" residuals: {em5}")

    y_model5 = z5 @ aCoeff5

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))

    data_list = [
        (m_data[:int_1], y_model1, aCoeff1, y1, m1, "0 < t < 35hrs"),
        (m_data[:int_2], y_model2, aCoeff2, y2, m2, "35hrs < t < 45hrs"),
        (m_data[:int_3], y_model3, aCoeff3, y3, m3, "45hrs < t < 73hrs"),
        (m_data[:int_4], y_model4, aCoeff4, y4, m4, "73hrs < t < 96hrs"),
        (m_data[:int_5], y_model5, aCoeff5, y5, m5, "96hrs < t < 120hrs"),
    ]

    for ax, (m_data, y_model, aCoeff, y_scatter, m_scatter, title) in zip(axes, data_list):
        ax.plot(m_scatter, y_model, label=f"y = {aCoeff[0]:.2f} + {aCoeff[1]:.2f}m")
        ax.scatter(m_scatter, y_scatter)
        ax.set_xlabel("m_data")
        ax.set_ylabel("y_model")
        ax.set_title(title)
        ax.legend()
        ax.grid()

    plt.tight_layout()
    plt.savefig("../figures/regression_lines.png")


if __name__ == "__main__":
    main()

