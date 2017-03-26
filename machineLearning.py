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

# machine learning solver
class Solver:
    def __init__(self, Lambda, Alpha, Iterations):
        self.Lambda = Lambda
        self.Alpha = Alpha
        self.Iterations = Iterations

    # set the constants
    def setConstants(self, Lambda, Alpha, Iterations):
        self.Lambda = Lambda
        self.Alpha = Alpha
        self.Iterations = Iterations

    # Sigmoid function
    def sigmoid(self, z):

        return sp.expit(z)

    # cost function for binary classification
    def costFunction(self, Theta, X, Y):
        m = len(Y); # number of training examples
        Theta = np.asmatrix(Theta).reshape((Theta.shape[0], 1))
        H = X * Theta # hypothesis
    
        # cost function
        J = - (np.transpose(Y) * np.log(self.sigmoid(H)) + np.transpose(1 - Y) * np.log(1 - self.sigmoid(H)))

        # introduce regularization
        if self.Lambda != 0:
            J = J + (self.Lambda / 2) * np.sum(np.power(Theta, 2))

        return J * (m**(-1))

    # gradient 
    def gradient(self, Theta, X, Y):
        grad = mtl.zeros((len(Theta), 1)) # gradient vector
        Theta = np.asmatrix(Theta).reshape((Theta.shape[0], 1))
        m = len(Y); # number of training examples
        H = X * Theta # hypothesis

        # calculate gradient
        grad = np.transpose(X) * (self.sigmoid(H) - Y) + self.Lambda * Theta
        grad[0] = grad[0] - self.Lambda * Theta[0] # Theta_0 not regularized

        return grad * (m**(-1))

    # gradient descent
    def gradDescent(self, Theta, X, Y, Plot):
        m = len(Y); # number of training examples
        J = mtl.zeros((self.Iterations, 1))  # cost function
        
        if Alpha != 0:
            for it in range(self.Iterations):  
                grad = self.gradient(Theta, X, Y) # gradient update          
                Theta = Theta - self.Alpha * grad
                if Plot:
                    J[it] = self.costFunction(Theta, X, Y)
        else: print('Please specify a value for Alpha that is different from 0')

        # monitor process
        if Plot:
            # plot cost function against the number of iterations
            plt.plot(J)
            plt.xlabel('Iterations')
            plt.title(r'Cost Function - $J(\theta)$')
            plt.show()

        # show accuracy on training set
        print 'Accuracy on training set using Gradient Descent: %.2f %%' %(self.accuracy(self.predict(Theta, X), Y))

        return Theta

    # advanced optimization (L-BFGS)
    def optTheta(self, Theta, X, Y):
        Theta = (opt.fmin_l_bfgs_b(costFunction, Theta, fprime = gradient, args=(X, Y, self.Lambda)))[0]
        Theta = np.asmatrix(Theta).reshape((len(initial_Theta), 1))
        # show accuracy on training set
        print 'Accuracy on training set using L-BFGS: %.2f %%' %(self.accuracy(self.predict(Theta, X), Y))

        return Theta

    # normal equation, not suggested with logistic regression
    def normEq(self, X, Y):
        n = X.shape[1] # number of variables
        K = mtl.zeros((n, n)) # correction matrix to perform regularization
        if self.Lambda != 0: 
            for i in range(n):
               for j in range(n):
                  if i != 0 and i == j:
                        K[i, j] = 1

        return np.linalg.pinv(np.transpose(X) * X + self.Lambda * K) * np.transpose(X) * Y

    # feature normalization
    def featNorm(self, X):
        m = X.shape[0] # number of training examples
        n = X.shape[1] # number of variables
        for i in range(1, n): # not applied to X_0 = 1
            mean = np.mean(X[:, i])
            diff = np.amax(X[:, i]) - np.amin(X[:, i])
            X[:, i] = (X[:, i] - mean) / diff

        return X

    # add features
    def addFeat(self, Theta, X, num):
        m = X.shape[0] # number of training examples
        n = X.shape[1] # number of variables
        new_X = mtl.zeros((m, n + num))
        new_Theta = mtl.zeros((n + num, 1))
        for j in range(m):
            for k in range(n):
                new_X[j, k] = X[j, k]
            for i in range(num):
                if i < n:
                    new_X[j, n + i] = np.power(X[j, i], 2) # x(i)^2     
                else: 
                    new_X[j, n + i] = X[j, i - n] * X[j, i + 1 - n] # x(i)*x(i+1)"""

        del Theta, X

        return new_Theta, new_X

    # output prediction
    def predict(self, Theta, X):
        H = X * Theta # hypothesis
        m = len(H) # number of training examples
        p = mtl.zeros((m, 1)) # prediction
        for i in range(m):
            if self.sigmoid(H[i]) >= 0.5:
                p[i] = 1
            else: p[i] = 0

        return p

    # accuracy on training set
    def accuracy(self, p, Y):
        m = len(Y) # number of training examples

        return (1 - np.sum(np.absolute(p - Y)) / m) * 100

    # split set into training, validation and test blocks
    def splitSets(self, X, Y, train_X, val_X, test_X):
        m = X.shape[0] # number of training examples
        n = X.shape[1] # number of variables
        rest = m % 10; # split sets by multiples of 10

        # training set
        X_train = np.delete(X, range(int((m - rest) * train_X + rest), m), 0) 
        Y_train = np.delete(Y, range(int((m - rest) * train_X + rest), m), 0) 
        # validation set
        X_val = np.delete(X, range(0, int((m - rest) * train_X + rest)), 0) 
        X_val = np.delete(X_val, range(int((m - rest) * val_X), m), 0) 
        Y_val = np.delete(Y, range(0, int((m - rest) * train_X + rest)), 0) 
        Y_val = np.delete(Y_val, range(int((m - rest) * val_X), m), 0)
        # test set
        X_test = np.delete(X, range(0, m - int((m - rest) * test_X)), 0) 
        Y_test = np.delete(Y, range(0, m - int((m - rest) * test_X)), 0) 

        del X, Y

        return X_train, Y_train, X_val, Y_val, X_test, Y_test

    # K-Fold Cross Validation 
    def crossVal(self, Theta, X, Y, K, algorithm):
        m_0 = X.shape[0] # number of training examples
        n = X.shape[1] # number of variables
        rest = m_0 % K; # split sets by multiples of K and delete the rest, if any
        X = np.delete(X, range(0, rest), 0)
        m = X.shape[0] # updated number of training examples

        for i in range(0, m, K):
            new_X = np.delete(X, range(0, i * K), 0) 
            new_X = np.delete(new_X, range((i + 1) * K, m), 0) 
            Y_val = np.delete(Y, range(0, i * K), 0)
            Y_val = np.delete(Y_val, range((i + 1) * K, m), 0) 
            X_eval = X[range(i * K, (i + 1) * K), :]

            if algorithm == 'optTheta':
                Theta = self.optTheta(Theta, new_X, new_Y)
                # show accuracy 
                print 'Accuracy on validation set using L-BFGS: %.2f %%' %(self.accuracy(self.predict(Theta, X_eval), Y))
            elif algorithm == 'gradDescent':
                Theta = self.gradDescent(Theta, new_X, new_Y, True)
                # show accuracy 
                print 'Accuracy on validation set using L-BFGS: %.2f %%' %(self.accuracy(self.predict(Theta, X_eval), Y))

            del new_X, new_Y, X_eval

