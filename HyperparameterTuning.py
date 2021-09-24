from math import cos
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy import optimize
import seaborn as sns
import numpy as np

def load_data():

    data = sio.loadmat("ex5data1.mat")
    
    return map(np.ravel,[data['X'],data['y'],data['Xval'],data['yval'],data['Xtest'],data['ytest']])

def cost(theta,X,y):

    m = X.shape[0]

    return ((X @ theta - y).T @ (X @ theta - y)) / (2 * m)

def gradient(theta,X,y):

    m = X.shape[0]

    return ( X.T @ (X @ theta - y) ) / m

def regularized_gradient(theta,X,y,l=1):
    
    m = X.shape[0]

    regularized_term = theta.copy()
    regularized_term[0] = 0

    regularized_term = (l/m) * regularized_term

    return gradient(theta,X,y) + regularized_term

def regularized_cost(theta,X,y,l=1):
    
    m = X.shape[0]

    regularized_term = (l / (2 * m))* np.power(theta[1:],2).sum()

    return cost(theta,X,y) + regularized_term

def regularized_cost(theta,X,y,l=1):

    m = X.shape[0]

    regularized_term = ( l / (2 * m) ) * np.power(theta[1:],2).sum()

    return cost(theta,X,y) + regularized_term

def linear_regression_np(X,y,l=1):

    theta = np.ones(X.shape[1])

    res = optimize.minimize(fun=regularized_cost,
                        x0 = theta,
                        args=(X,y,l),
                        method='TNC',
                        jac = regularized_gradient,
                        options={'disp':False})
    return res

def prepare_poly_data(*args,power):

    def prepare(x):

        df = poly_features(x,power=power)

        ndarr = normalize_feature(df).iloc[:,:].values

        return np.insert(ndarr,0,np.ones(ndarr.shape[0]),axis=1)

    return [prepare(x) for x in args]

def poly_features(x,power,as_ndarray=False):

    data = {'f{}'.format(i):np.power(x,i) for i in range(1 , power + 1)}
    df = pd.DataFrame(data)

    return df.as_matrix() if as_ndarray else df

def normalize_feature(df):

    return df.apply(lambda column:(column - column.mean()) / column.std())

def plot_learning_curve(X,y,Xval,yval,l=0):

    training_cost ,cv_cost = [] , []
    m = X.shape[0]

    for i in range(1 , m+1):

        res = linear_regression_np(X[:i,:],y[:i],l=l)

        tc = cost(res.x,X[:i,:],y[:i])
        cv = cost(res.x,Xval,yval)

        training_cost.append(tc)
        cv_cost.append(cv)

    plt.plot(np.arange(1, m + 1), training_cost, label='training cost')
    plt.plot(np.arange(1, m + 1), cv_cost, label='cv cost')
    plt.legend(loc=1)

if __name__ == '__main__':
    X , y , Xval , yval , Xtest , ytest = load_data()
    #"""
    df = pd.DataFrame({'water_level':X,'flow':y})
    sns.lmplot('water_level','flow',data=df,fit_reg=False,height=5)
    plt.show()
    #"""

    X_poly,Xval_poly,Xtest_poly = prepare_poly_data(X,Xval,Xtest,power=8)
    plot_learning_curve(X_poly, y, Xval_poly, yval, l=0)
    plt.show()
    #"""
    l_candidate = [0,0.003,0.006,0.009,0.03,0.06,0.09,0.3,0.6,0.9,3,6,9]
    training_cost,cv_cost = [],[]

    for l in l_candidate:
        res = linear_regression_np(X_poly,y,l)

        tc = cost(res.x,X_poly,y)
        cv = cost(res.x,Xval_poly,yval)

        training_cost.append(tc)
        cv_cost.append(cv)
    plt.plot(l_candidate, training_cost, label='training')
    plt.plot(l_candidate, cv_cost, label='cross validation')
    plt.legend(loc=2)
    plt.xlabel('lambda')
    plt.ylabel('cost')
    plt.show()

    for l in l_candidate:
        theta = linear_regression_np(X_poly, y, l).x
        print('test cost(l={}) = {}'.format(l, cost(theta, Xtest_poly, ytest)))
    #"""
