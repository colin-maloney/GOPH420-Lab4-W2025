import numpy as np
import matplotlib.pyplot as plt


def multiregression(y, Z):
    """
    preform multiple linear regression
    ----------
    parameters:
    y : arraylike, shape (n, ) or (n,1)
        the vector of dependent variables
    Z : arraylike, shape (n, m)
        the matrix of independent variables

    returns:
    ----------
    numpy.ndarray, shape (m, ) or (m,1)
        the vector of regression coefficients
    numpy.ndarray, shape (n, ) or (n,1)
        the vector of residuals
    float
        the coefficient of determination (R^2)
    """
    y = np.array(y)
    Z = np.array(Z)

    y_mean = np.average(y)
    aCoeff = np.linalg.inv(np.transpose(Z)*Z) * (np.transpose(Z)*y)

    ey = []
    for i, yi in enumerate(y):
        ey.append(yi - y_mean) 
    Sy = np.sum(ey**2)  
    
    ym = Z * aCoeff

    ey = []
    for i, yi in enumerate(y):
        ey.append(yi - ym[i]) 
    Sr = np.sum(ey**2) 

    R = (Sy - Sr) / Sy 

    return aCoeff, ey, R


