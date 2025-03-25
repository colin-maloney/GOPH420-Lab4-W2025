import numpy as np 
import matpotlib.pyplot as plt 

def multiregression(): 
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
    