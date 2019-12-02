# import pandas as pd
import numpy as np

def costFunc(X,Y,theta):
    inner = np.power((X*theta.T)-Y,2)
    return np.sum(inner)/(2*len(X))

def gradientDescent(X,Y,theta,alpha,iters):
    temp = np.mat(np.zeros(theta.shape))
    cost = np.zeros(iters)
    thetaNums = int(theta.shape[1])
    print(thetaNums)
    for i in range(iters):
        error = (X*theta.T-Y)
        for j in range(thetaNums):
            derivativeInner = np.multiply(error,X[:,j])
            temp[0,j] = theta[0,j] - (alpha*np.sum(derivativeInner)/len(X))

        theta = temp
        cost[i] = costFunc(X,Y,theta)

    return theta,cost


