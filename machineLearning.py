########################## MACHINE LEARNING ##########################
""" FUNCTIONS:
1. Sigmoid, as the probability for the event to occur
2. binCostFunction, to monitor the cost function:
    1) Theta: parameters vector
    2) X, Y: variables and outputs vectors
    3) Lambda: regularization parameter
3. gradientDescent, to optimise Theta:
    1) Theta, X, Y, Lambda: as in binCostFunction
    2) Alpha: learning rate
    3) Iterations: number of iterations
    4) Monitor: boolean variable. If True, the cost function is plotted 
        against the number of iterations to monitor convergence,
        and the accuracy on the training set is displayed
4. prediction, to predict the output basing on the model obtained after learning.
    Returns 1 if the hypothesis model results in a value >= 0.5, 0 otherwise:
    1) Theta, X: as in binCostFunction
NOTE: make sure to include the bias column in matrix X, 
    before passing the latter to any function. """

import numpy as np
import numpy.matlib as mtl
import matplotlib.pyplot as plt
import scipy.optimize as opt
import scipy.special as sp

# Sigmoid function
def sigmoid(z):

    return sp.expit(z)

# cost function for binary classification
def costFunction(Theta, X, Y, Lambda):
    m = len(Y); # number of training examples
    Theta = np.asmatrix(Theta).reshape((Theta.shape[0], 1))
    H = X * Theta # hypothesis
    
    # cost function
    J = - (np.transpose(Y) * np.log(sigmoid(H)) + np.transpose(1 - Y) * np.log(1 - sigmoid(H)))

    # introduce regularization
    if Lambda != 0:
        J = J + (Lambda / 2) * np.sum(np.power(Theta, 2))

    return J * (m**(-1))

# gradient 
def gradient(Theta, X, Y, Lambda):
    grad = mtl.zeros((len(Theta), 1)) # gradient vector
    Theta = np.asmatrix(Theta).reshape((Theta.shape[0], 1))
    m = len(Y); # number of training examples
    H = X * Theta # hypothesis

    # calculate gradient
    grad = np.transpose(X) * (sigmoid(H) - Y) + Lambda * Theta
    grad[0] = grad[0] - Lambda * Theta[0] # Theta_0 not regularized

    return grad * (m**(-1))

# gradient descent
def gradDescent(Theta, X, Y, Alpha, Lambda, Iterations, Plot):
    m = len(Y); # number of training examples
    J = mtl.zeros((Iterations, 1))  # cost function
    
    if Alpha != 0:
        for it in range(Iterations):  
            grad = gradient(Theta, X, Y, Lambda) # gradient update          
            Theta = Theta - Alpha * grad
            if Plot:
                J[it] = costFunction(Theta, X, Y, Lambda)
    else: print('Please specify a value for Alpha that is different from 0')

    # monitor process
    if Plot:
        # plot cost function against the number of iterations
        plt.plot(J)
        plt.xlabel('Iterations')
        plt.title(r'Cost Function - $J(\theta)$')
        plt.show()

    # show accuracy on training set
    print 'Accuracy on training set using Gradient Descent: %.2f %%' %(accuracy(predict(Theta, X), Y))

    return Theta

# advanced optimization (L-BFGS)
def optTheta(initial_Theta, X, Y, Lambda):
    Theta = (opt.fmin_l_bfgs_b(costFunction, initial_Theta, fprime = gradient, args=(X, Y, Lambda)))[0]
    Theta = np.asmatrix(Theta).reshape((len(initial_Theta), 1))
    # show accuracy on training set
    print 'Accuracy on training set using L-BFGS: %.2f %%' %(accuracy(predict(Theta, X), Y))

    return Theta

# normal equation, not suggested with logistic regression
def normEq(X, Y, Lambda):
    n = X.shape[1] # number of variables
    K = mtl.zeros((n, n)) # correction matrix to perform regularization
    if Lambda != 0: 
        for i in range(n):
           for j in range(n):
              if i != 0 and i == j:
                    K[i, j] = 1

    return np.linalg.pinv(np.transpose(X) * X + Lambda * K) * np.transpose(X) * Y

# feature normalization
def featNorm(X):
    n = X.shape[1] # number of variables
    m = X.shape[0] # number of training examples
    for i in range(1, n): # not applied to X_0 = 1
        mean = np.mean(X[:, i])
        diff = np.amax(X[:, i]) - np.amin(X[:, i])

        for j in range(m):
            X[j, i] = (X[j, i] - mean) / diff

    return X

# output prediction
def predict(Theta, X):
    H = X * Theta # hypothesis
    m = len(H) # number of training examples
    p = mtl.zeros((m, 1)) # prediction
    for i in range(m):
        if sigmoid(H[i]) >= 0.5:
            p[i] = 1
        else: p[i] = 0

    return p

# accuracy on training set
def accuracy(p, Y):
    m = len(Y) # number of training examples

    return (1 - np.sum(np.absolute(p - Y)) / m) * 100