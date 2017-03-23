""" Logistic Regression for Binary Classification Machine Learning problems
FUNCTIONS:
1. Sigmoid, as the probability for the event to occur.
    If the argument overcomes 100 in absolute value, then 100 is used
    (no change in result)
2. binCostFunction, to monitor the cost function:
    1) Theta: parameters vector
    2) X, Y: variables and outputs vectors
    3) Lambda: regularization parameter
3. gradientDescent, to optimise Theta:
    1) Theta, X, Y, Lambda: as in binCostFunction
    2) Alpha: learning rate
    3) Iterations: number of iterations
    4) Plot: boolean variable. If True,
        the cost function is plotted against the number of iterations,
        to monitor convergence
4. prediction, to predict the output basing on the model obtained after learning.
    Returns 1 if the hypothesis model results in a value >= 0.5, 0 otherwise """

import numpy as np
from numpy import matlib
from matplotlib import pyplot as plt

# Sigmoid function
def sigmoid(z):
    if np.absolute(z) > 100:
        # return the correct sign and change value to 100
        z = 100 * (z / np.absolute(z))

    sig = 1 / (1 + np.power(np.e, -z))
    if sig == 1:
        # avoid singularity
        sig = 0.999

    return sig

# cost function for binary classification
def binCostFunction(Theta, X, Y, Lambda):
    m = len(Y); # number of training examples
    H = X * Theta # hypothesis

    sig = np.matlib.zeros((m, 1))
    for i in range(m):
        sig[i] = sigmoid(H[i])

    # cost function
    J = - (np.transpose(Y) * np.log(sig) + np.transpose((np.ones((m, 1)) - Y)) * np.log(np.ones((m, 1)) - sig))

    # introduce regularization
    if Lambda != 0:
        J = J + (Lambda / 2) * np.sum(np.power(Theta, 2))

    return J * (m**(-1))

# gradient descent
def gradDescent(Theta, X, Y, Alpha, Lambda, Iterations, Plot):
    m = len(Y); # number of training examples
    J = np.matlib.zeros((Iterations, 1))  # cost function

    if Alpha != 0:
        for it in range(Iterations):
            H = X * Theta # hypothesis
            sig = np.matlib.zeros((m, 1))
            for i in range(m):
                sig[i] = sigmoid(H[i])

            for j in range(len(Theta)):
                # calculate gradient
                if j == 0:
                    gradient = np.transpose(sig - Y) * X[:, j]
                else:
                    gradient = np.transpose(sig - Y) * X[:, j] + Lambda * Theta[j]
                
                Theta[j] = Theta[j] - Alpha * (m**(-1)) * gradient
            if Plot:
                J[it, 0] = binCostFunction(Theta, X, Y, Lambda)
    else: print('Please specify a value for Alpha that is different from 0')
    # plot cost function against the number of iterations
    if Plot:
        plt.plot(J)
        plt.xlabel('Iterations')
        plt.ylabel(r'Cost Function - $J(\theta)$')
        plt.show()

    return Theta

# normal equation
def normEq(X, Y, Lambda):
    n = X.shape[1]
    K = np.matlib.zeros((n, n)) # correction matrix to perform regularization
    if Lambda != 0: 
        for i in range(n):
           for j in range(n):
              if i != 0 and i == j:
                    K[i, j] = 1

    return np.linalg.pinv(np.transpose(X) * X + Lambda * K) * np.transpose(X) * Y

# output prediction
def predict(Theta, X, Alg):
    H = X * Theta # hypothesis
    m = len(H) # number of training examples
    p = np.matlib.zeros((m, 1)) # prediction

    for i in range(m):
        if Alg == 'normEq':
            var = H[i]
        elif Alg == 'gradDescent':
            var = sigmoid(H[i])
        else: print('Please pick an algorithm between normEq and gradDescent')

        if var >= 0.5:
            p[i] = 1
        else: p[i] = 0

    return p
