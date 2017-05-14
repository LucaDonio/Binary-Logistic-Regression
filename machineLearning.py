############################################ MACHINE LEARNING ############################################

import numpy as np
import numpy.matlib as mtl
import matplotlib.pyplot as plt
import scipy.optimize as opt
import scipy.special as sp

# machine learning solver
class Solver:

    ############################# PRE-PROCESSING, VALIDATION AND OPTIMISATION #############################

    # add bias column in X
    def addBias(self, X):
        X = np.insert(X, 0, 1.0, axis = 1)

        return X

    # feature normalization
    def featNorm(self, X):
        m = X.shape[0] # number of training examples
        n = X.shape[1] # number of variables
        new_X = mtl.ones((m, n)) # all ones so the bias unit remains equal to 1.0
        for i in xrange(1, n): # not applied to X_0 = 1.0
            mean = float(np.mean(X[:, i]))
            diff = float(np.amax(X[:, i]) - np.amin(X[:, i]))
            # avoid singularity
            if diff == 0.0:
                new_diff = 1.0
            else: new_diff = diff
            new_X[:, i] = (X[:, i] - mean) / new_diff

        return new_X

    # add features
    def addFeat(self, X, num):
        m = X.shape[0] # number of training examples
        n = X.shape[1] # number of variables
        new_X = mtl.zeros((m, n + num))
        for j in xrange(m):
            for k in xrange(n):
                new_X[j, k] = X[j, k]
            for i in xrange(num):
                if i < n:
                    new_X[j, n + i] = np.power(X[j, i], 2) # x(i)^2     
                else: 
                    new_X[j, n + i] = X[j, i - n] * X[j, i + 1 - n] # x(i)*x(i+1)"""

        return new_X

    # modify Y from vector of lables (i.e. [1 2 3 4 ...]) to binary classification matrix (i.e. [[1 0 0 0 ...], [0 1 0 0 ...], ...])
    def YtoBin(self, Y, number_of_labels):
        m = Y.shape[0] # number of training examples
        new_Y = mtl.zeros((m, number_of_labels))
        for i in xrange(m):
            for label in xrange(number_of_labels):
                if Y[i, 0] == label:
                    new_Y[i, label] = 1.0

        return new_Y

    # split set into training, validation and test blocks
    def splitSets(self, X, Y, m_train, m_val, m_test):
        m = X.shape[0] # number of training examples
        n = X.shape[1] # number of variables

        # training set
        X_train = X[xrange(m_train)]
        Y_train = Y[xrange(m_train)]
        # validation set
        X_val = X[xrange(m_train, m_train + m_val)]
        Y_val = Y[xrange(m_train, m_train + m_val)]
        # test set
        X_test = X[xrange(m_train + m_val, m_train + m_val + m_test)]
        Y_test = Y[xrange(m_train + m_val, m_train + m_val + m_test)]

        return X_train, Y_train, X_val, Y_val, X_test, Y_test

    # unroll list
    def unrollList(self, lst):
        lst_unr = []
        for i in xrange(len(lst)):
            lst[i] = lst[i].ravel()
            lst[i] = np.asmatrix(lst[i])
            lst[i] = lst[i].reshape((lst[i].shape[1], 1))
            for j in xrange(lst[i].shape[0]):
                lst_unr.append(lst[i][j])
        lst = mtl.zeros((len(lst_unr), 1)) 
        for i in xrange(len(lst_unr)):
            lst[i] = lst_unr[i]

        return lst

    # Sigmoid function
    def sigmoid(self, z):
        sig = sp.expit(z)

        # avoid singularity in logarithm
        e = 10**(-10)
        new_sig = mtl.zeros((len(sig), 1))
        for i in xrange(len(sig)):
            if sig[i] == 0.0:
                new_sig[i] = sig[i] + e
            elif sig[i] == 1.0:
                new_sig[i] = sig[i] - e
            else:
                new_sig[i] = sig[i]  
        del sig

        return new_sig

    # Sigmoid function gradient
    def sigmoidGrad(self, z):
        sig_grad = mtl.zeros((len(z), 1))
        for i in xrange(len(z)):
            sig_grad[i] = self.sigmoid(z[i]) * (1 - self.sigmoid(z[i]))

        return sig_grad

    # Gaussian kernel
    def gaussKernel(self, X_1, X_2, sigma):
        X_1 = np.asmatrix(X_1).reshape((len(X_1), 1))
        X_2 = np.asmatrix(X_2).reshape((len(X_2), 1))

        num = float(np.sum(np.power(X_1 - X_2, 2)))
        den = float(2 * np.power(sigma, 2))

        return np.exp(- num / den)

    # normal equation (only suggested with linear regression)
    def normEq(self, X, Y):
        n = X.shape[1] # number of variables
        K = mtl.zeros((n, n)) # correction matrix to perform regularization
        if self.Lambda != 0: 
            for i in xrange(n):
               for j in xrange(n):
                  if i != 0 and i == j:
                        K[i, j] = 1

        return np.linalg.pinv(np.transpose(X) * X + self.Lambda * K) * np.transpose(X) * Y

    # model accuracy
    def accuracy(self, p, Y):
        m = Y.shape[0] # number of training examples
        count = 0 # true values count
        for i in xrange(m):
            if p[i] == Y[i]:
                count += 1

        return 100 * (count / m)

    # plot the cost function to find an appropriate value for Alpha
    def findAlpha(self, model, X, Y, Alpha_vec):
        m = Y.shape[0]; # number of training examples
        J = mtl.zeros((model.Iterations, 1))  # cost function

        # reshape target vector (YtoBin), unless it is already in binary format (1 or 0)
        if model.output_size > 1:
            Y = self.YtoBin(Y, model.output_size)

        Theta = model.initialiseTheta() # initialise Theta

        print 'Calculating cost functions: J(Alpha)...'
        for i in xrange(len(Alpha_vec)):
            model.Alpha = Alpha_vec[i]
            for it in xrange(model.Iterations):
                grad = model.gradient(Theta, X, Y) # gradient update          
                Theta = Theta - model.Alpha * grad
                J[it] = model.costFunction(Theta, X, Y)

            print 'Alpha = ' + str(Alpha_vec[i]) + ' done!'
            # plot the cost function against number of iterations, for each Alpha
            plt.plot(J, label = r'$\alpha$ = ' + str(Alpha_vec[i]))
            plt.xlabel('Iterations')
            plt.title(r'Cost Function - $J(\theta)$')
            plt.legend()
            
        plt.show()

    # find optimal Lambda that minimises validation error
    def findLambda(self, model, X_train, Y_train, X_val, Y_val, Lambda_vec, algorithm = 'lbfgs', Plot = False):
        Error_val = [] # validation error values

        # reshape target vector (YtoBin), unless it is already in binary format (1 or 0)
        if model.output_size > 1:
            Y_train = self.YtoBin(Y_train, model.output_size)
            Y_val = self.YtoBin(Y_val, model.output_size)

        Theta = model.initialiseTheta() # initialise Theta

        print 'Calculating errors...'
        for i in xrange(len(Lambda_vec)):
            print 'Lambda = ' + str(Lambda_vec[i]) + ':'
            model.Lambda = Lambda_vec[i]

            # train the model          
            if algorithm == 'lbfgs':
                Theta = (opt.fmin_l_bfgs_b(model.costFunction, Theta, fprime = model.gradient, args = (X_train, Y_train), maxiter = model.Iterations))[0]
            elif algorithm == 'gd':
                for it in xrange(model.Iterations):
                    grad = model.gradient(Theta, X, Y) # gradient update          
                    Theta = Theta - model.Alpha * grad
            
            #if self.Model == 'logR':
                #Theta = Theta.reshape((Theta.shape[0], Theta.shape[1]))

            # evaluate validation error (cost function) 
            model.Lambda = 0.0
            valErr = model.costFunction(Theta, X_val, Y_val)
            Error_val.append(valErr)
            print 'Validation error: ' + str(valErr)

        # Lambda that minimizes error
        model.Lambda = Lambda_vec[np.argmin(Error_val)]

        print 'Regularization parameter that minimizes the validation error: ' + str(model.Lambda)
        print 'Lambda was set equal to ' + str(model.Lambda)

        if Plot:
            # plot errors against Lambda
            plt.plot(Lambda_vec, Error_val)
            plt.xlabel('$\lambda$')
            plt.title(r'Validation Error - $J_{val}(\theta)$')
            plt.show()

    # plot learning curves
    def plotLearningCurve(self, model, X_train, Y_train, X_val, Y_val, algorithm = 'lbfgs', m_max = 100, step = 10):
        Error_val = [] # validation error values
        Error_train = [] # training error values
        training_lambda = model.Lambda # retain Lambda for training

        # reshape target vector (YtoBin), unless it is already in binary format (1 or 0)
        if model.output_size > 1:
            Y_train = self.YtoBin(Y_train, model.output_size)
            Y_val = self.YtoBin(Y_val, model.output_size)

        Theta = model.initialiseTheta() # initialise Theta

        if m_max <= X_train.shape[0]:
            print 'Calculating errors...'
            if m_max >= 100:
                print 'This may take some time... crack open a cold one!'
            for i in xrange(1, m_max + 1, step):
                # training    
                model.Lambda = training_lambda     
                if algorithm == 'lbfgs':
                    Theta = (opt.fmin_l_bfgs_b(model.costFunction, Theta, fprime = model.gradient, args = (X_train[xrange(0, i)], Y_train[xrange(0, i)]), maxiter = model.Iterations))[0]
                elif algorithm == 'gd':
                    for it in xrange(model.Iterations):
                        grad = model.gradient(Theta, X_train[xrange(0, i)], Y_train[xrange(0, i)]) # gradient update          
                        Theta = Theta - model.Alpha * grad

                #if self.Model == 'logR':
                    #Theta = Theta.reshape((Theta.shape[0], Theta.shape[1]))

                # evaluate errors (cost function) 
                model.Lambda = 0.0 # set Lambda equal to zero
                valErr = model.costFunction(Theta, X_val, Y_val)
                trainErr = model.costFunction(Theta, X_train[xrange(0, i)], Y_train[xrange(0, i)])
                Error_val.append(valErr)
                Error_train.append(trainErr)
                print 'm = ' + str(i)
                print 'Validation error: ' + str(valErr)
                print 'Training error: ' + str(trainErr)

            # plot errors against number of training examples
            plt.plot(Error_val, label = r'Validation Error')
            plt.plot(Error_train, label = r'Training Error')
            plt.xlabel('Number of training examples')
            plt.title(r'Learning Curves - $J_{val}(\theta)$')
            plt.legend()
            plt.show()

        else: print 'The maximum number of training examples exceeds the number or training examples available. Pick a lower m_max'

    # K-fold Cross Validation to evaluate model's performance
    def crossVal(self, K, initial_Theta, X, Y, algorithm):
        m = X.shape[0] # number of training examples
        n = X.shape[1] # number of variables
        rest = m % K # rest to K divisions
        X_new = np.delete(X, xrange(rest), 0) # get rid of rest-data
        Y_new = np.delete(Y, xrange(rest), 0) 
        m_new = m - rest
        block = m_new / K
        Accuracies = []

        print 'K-fold cross validation started...'
        for i in xrange(K):
            X_train = np.delete(X_new, xrange(i * block, (i + 1) * block), 0) 
            X_test = X_new[xrange(i * block, (i + 1) * block)]
            Y_train = np.delete(Y_new, xrange(i * block, (i + 1) * block), 0) 
            Y_test = Y_new[xrange(i * block, (i + 1) * block)]

            # train the model
            if algorithm == 'lbfgs':
                Theta = (opt.fmin_l_bfgs_b(self.costFunction, initial_Theta, fprime = self.gradient, args = (X_train, Y_train), maxiter = self.Iterations))[0]
            elif algorithm == 'gd':
                Theta = self.gradDescent(initial_Theta, X_train, Y_train, False)
            Theta = np.asmatrix(Theta).reshape((Theta.shape[0], Theta.shape[1]))

            # calculate accuracy  
            acc = (self.performance(self.predict(Theta, X_test), Y_test))[0]
            Accuracies.append(acc)
            print 'Accuracy on test set (%.0f/%.0f): %.2f %%' %(i + 1, K, acc)

        print 'Mean accuracy after K-fold cross validation: %.2f %%' %(np.mean(np.asmatrix(Accuracies).reshape((K, 1))))

    ########################################## NEURAL NETWORK ##########################################
    class NeuralNetwork:

        # Neural Network initialisation
        def __init__(self, X, Y, hidden_sizes = [0], initialisation = 'rand', Alpha = 0.01, Lambda = 0.0, Iterations = 1e5):
            # hidden_sizes, if exists, is a vector containing the size of each hidden layer (e.g. [3, 5, 6, ...])
            # its default value is [0], meaning no hidden layer is used
            if hidden_sizes == [0]:
                len_hidden = 0
            else: len_hidden = len(hidden_sizes)
            if np.max(Y) == 1.0: # binary output (1 or 0)
                self.output_size = 1 
            else: self.output_size = int(np.max(Y) + 1) # number of output labels
            self.input_size = int(X.shape[1] - 1) # number of input features, deducting the bias
            self.layers_number = len_hidden + 2 # number of layers (hidden + input + output)
            self.layers_sizes = [] # vector containing sizes of all layers
            self.layers_sizes.append(self.input_size)
            for i in xrange(len_hidden):
                self.layers_sizes.append(hidden_sizes[i])
            self.layers_sizes.append(self.output_size)
            self.initialisation = initialisation # Theta initialisation method

            # target vector
            self.Y = Y
            # trainig examples 
            self.X = X

            # constants
            # regularization parameter
            self.Lambda = Lambda
            # learning rate
            self.Alpha = Alpha
            # number of iterations
            self.Iterations = Iterations

            # inherit methods
            self.YtoBin = Solver().YtoBin
            self.unrollList = Solver().unrollList
            self.sigmoid = Solver().sigmoid
            self.sigmoidGrad = Solver().sigmoidGrad
            self.accuracy = Solver().accuracy

        # generate random values in initial Theta for Neural Network (symmetry breaking)
        # Theta dimensions: 
        # number of rows: next layer number of units
        # number of columns: previous layer number of units + 1 (bias)
        def initialiseTheta(self):
            inTheta = []
            for i in xrange(1, len(self.layers_sizes)):
                previous_layer_units = self.layers_sizes[i - 1]
                next_layer_units = self.layers_sizes[i]
                if self.initialisation == 'rand': # ramdon initialisation
                    # inTheta[i] in (-e, +e)
                    e_init = np.sqrt(6) / (np.sqrt(next_layer_units) + np.sqrt(previous_layer_units))
                    inTheta.append(np.random.rand(next_layer_units, previous_layer_units + 1) * 2 * e_init - e_init)
                elif self.initialisation == 'Xav': # Xavier's initialisation
                    Xav_w = 1.0 / previous_layer_units
                    inTheta.append(np.full((next_layer_units, previous_layer_units + 1), Xav_w))

            # unroll initial Theta
            inTheta = self.unrollList(inTheta)
            # inTheta is a list of matrices
            return inTheta

        # reshape Theta once it is unrolled
        def reshapeTheta(self, Theta):
            new_Theta = []
            next_idx = 0
            for i in xrange(1, len(self.layers_sizes)):
                previous_layer_units = self.layers_sizes[i - 1]
                next_layer_units = self.layers_sizes[i]
                Theta_layer = mtl.zeros((next_layer_units, previous_layer_units + 1))
                for j in xrange(next_layer_units):
                    for k in xrange(previous_layer_units + 1):
                        Theta_layer[j, k] = Theta[next_idx]
                        next_idx = next_idx + 1
                new_Theta.append(Theta_layer)

            return new_Theta

        # feedforward pass for Neural Network (returns activation functions and hypothesis for all layers)
        def feedForward(self, Theta, training_example):
            a = [] # activation functions
            z = [] # hypothesis
            a.append(np.transpose(training_example)) # input layer contains bias unit  
            a_prec = np.asmatrix(a[0])
            # hidden and output layers
            for j in xrange(1, self.layers_number): 
                # Theta is a list of matrices, one for each layer jump
                Theta_layer = np.asmatrix(Theta[j - 1])
                # layer hypothesis 
                z.append(Theta_layer * a_prec)
                z_temp = np.asmatrix(z[j - 1])
                # layer activation function               
                a_prec = np.asmatrix(self.sigmoid(z_temp))
                # add bias unit if not output layer
                if j != self.layers_number - 1:
                    a_prec = np.insert(a_prec, 0, 1.0, axis = 0) 
                a.append(a_prec)

            return a, z

        # Backpropagation to calculate gradients in Neural Networks: returns the errors vector
        # note that Y must be converted to binary vectors first (YtoBin function)
        def backPropagationError(self, Theta, a, z, training_example_label):
            delta = [] # errors
            delta.append(a[len(a) - 1] - np.transpose(training_example_label)) # output layer
            delta_prec = np.asmatrix(delta[0])
            # hidden layers (no error in input layer)
            for j in xrange(self.layers_number - 2, 0, -1): # reversed loop (backpropagation: delta = [d10, d9, d8, ...])
                # Theta is a list of matrices, one for each layer jump
                Theta_layer = np.asmatrix(Theta[j])
                # layer error
                sig_grad = self.sigmoidGrad(z[j - 1])
                sig_grad = np.insert(sig_grad, 0, 1.0, axis = 0) # add bias for calculation
                delta_prec = np.multiply(np.transpose(Theta_layer) * delta_prec, sig_grad)
                delta_prec = np.delete(delta_prec, xrange(0, 1), 0) # delete bias
                delta.append(delta_prec)
            delta = delta[::-1] # reverse list (delta = [d1, d1, d3, ...])

            return delta

        def costFunction(self, Theta, X, Y):
            m = X.shape[0] # number of training examples

            # reshape Theta
            Theta = self.reshapeTheta(Theta)
            # cost function initialisation
            J_vec = []
            for i in xrange(m):
                # perform feed forward to calculate cost function
                a = self.feedForward(Theta, X[i, :])[0]
                J_temp = Y[i, :] * np.log(a[self.layers_number - 1]) + (1 - Y[i, :]) * np.log(1 - a[self.layers_number - 1])
                J_vec.append(J_temp)

            # calculate cost function
            J = - np.sum(J_vec) 

            # introduce regularization (Theta_0 is not regularized)
            if self.Lambda != 0.0:
                for j in xrange(len(Theta)): # for every layer jump
                    Theta[j] = np.asmatrix(Theta[j])
                    # first column not regularised
                    Theta_reg = np.insert(Theta[j][:, xrange(1, Theta[j].shape[1])], 0, 0.0, axis = 1) 
                    J += (self.Lambda / 2.0) * np.sum(np.power(Theta_reg, 2))

            return J * (1.0 / m)

        # gradient calculation
        def gradient(self, Theta, X, Y):
            m = X.shape[0] # number of training examples

            # reshape Theta
            Theta = self.reshapeTheta(Theta)
            # gradient initialisation
            grad = []
            for i in xrange(self.layers_number - 1):
                # Theta is a list of matrices, one for each layer jump
                Theta_layer = np.asmatrix(Theta[i])
                grad.append(mtl.zeros((Theta_layer.shape[0], Theta_layer.shape[1])))
            for i in xrange(m):
                a, z = self.feedForward(Theta, X[i, :])
                # perform back propagation to calculate gradients
                delta = self.backPropagationError(Theta, a, z, Y[i, :])
                for j in xrange(len(Theta)): # for every layer jump
                    Theta[j] = np.asmatrix(Theta[j])
                    # first column not regularised
                    Theta_reg = np.insert(Theta[j][:, xrange(1, Theta[j].shape[1])], 0, 0.0, axis = 1) 
                    grad[j] += (np.asmatrix(delta[j] * np.transpose(a[j])) + self.Lambda * Theta_reg) 

            # unroll gradients
            grad = self.unrollList(grad)

            return grad * (1.0 / m)

        # perform gradient checking for Neural Networks
        def gradChecking(self, Theta, X, Y):
            print 'Checking gradients...'
            Theta = self.unrollList(Theta)
            epsilon = 10**(-4)
            # back propagation gradients
            gradient_back_propagation = self.gradient(Theta, X, Y)
            # calculate numerical gradients
            grad = []
            for i in xrange(Theta.shape[0]):
                Theta_epsilon = mtl.zeros((Theta.shape[0], 1))
                Theta_epsilon[i] = epsilon
                Theta_plus = Theta + Theta_epsilon
                Theta_minus = Theta - Theta_epsilon
                # generate gradients
                grad.append((self.costFunction(Theta_plus, X, Y) - self.costFunction(Theta_minus, X, Y)) / (2 * epsilon))
            grad = np.asmatrix(grad)
            grad = grad.reshape((grad.shape[1], 1))
            # compute difference
            diff = np.linalg.norm(grad - gradient_back_propagation) / np.linalg.norm(grad + gradient_back_propagation)

            print 'Relative difference between numerical method and Back-Propagation: %.15e' %(diff)

        # train the Neural Network
        def trainNeuralNetwork(self, Algorithm = 'lbfgs'):
            # initialise Theta
            Theta = self.initialiseTheta()
            # reshape target vector (YtoBin), unless it is already in binary format (1 or 0)
            if self.output_size > 1:
                Y = self.YtoBin(self.Y, self.output_size)
            else: Y = self.Y
            print 'Training...'
            # lbfgs algorithm
            if Algorithm == 'lbfgs':
                alg = 'L-BFGS'
                # calculate weights
                Theta = (opt.fmin_l_bfgs_b(self.costFunction, Theta, fprime = self.gradient, args = (self.X, Y), maxiter = self.Iterations))[0]

            # gradient descent
            elif Algorithm == 'gd':
                alg = 'Gradient Decsent'
                m = Y.shape[0] # number of training examples
                J = mtl.zeros((self.Iterations, 1))  # cost functions
                if self.Alpha != 0:
                    for it in xrange(self.Iterations):  
                        J[it] = self.costFunction(Theta, self.X, Y) # cost function update
                        grad = self.gradient(Theta, self.X, Y) # gradient update
                        # calculate weights          
                        Theta = Theta - self.Alpha * grad

                else: print('Please specify a value for Alpha that is different from 0.')

                # monitor process: plot cost functions against iterations
                plt.plot(J, label = r'$\alpha$ = ' + str(self.Alpha))
                plt.xlabel('Iterations')
                plt.title(r'Cost Function - $J(\theta)$')
                plt.legend()
                plt.show()

            else:
                print 'The algorithm selected for optimisation was not identified: ' + Algorithm
                print 'Use "lbfgs" or "gd" instead.'

            # reshape Theta
            Theta = self.reshapeTheta(Theta)
            print 'Neural Network trained!'
            # show accuracy on training set
            print 'Accuracy on training set using ' + alg + ': %.2f %%' %(self.accuracy(self.predict(Theta, self.X), self.Y))

            return Theta

        # output prediction
        def predict(self, Theta, X):
            m = X.shape[0] # number of training examples
            p = [] # predictions
            for i in xrange(m):
                a = self.feedForward(Theta, X[i, :])[0] # hypothesis
                if self.output_size > 1: # if labels are > 1, calculate the highest probability amongst them
                    a_p = np.asmatrix(a[len(a) - 1])
                    a_p_max_idx = np.argmax(a_p)
                    a_p_max = np.max(a_p)
                    p.append(a_p_max_idx)
                else: # if only 1 label, return 1 if probability is > 0.5
                    if a[len(a) - 1] >= 0.5:
                        p.append(1)
                    else: p.append(0)

            # reshape prediction vector
            p = np.asmatrix(p).reshape((len(p), 1))

            return p

        # model testing
        def testModel(self, Theta, X_test, Y_test):

            print 'Accuracy on test set using Neural Network: %.2f %%' %(self.accuracy(self.predict(Theta, X_test), Y_test))

    ########################################## LOGISTIC REGRESSION ##########################################
    class LogisticClassifier:

        # Logistic Classifier initialisation
        def __init__(self, X, Y):
            self.output_size = int(Y.shape[1]) # number of output labels

            #Constants
            # regularization parameter
            self.Lambda = Solver().Lambda
            # learning rate
            self.Alpha = Solver().Alpha
            # number of iterations
            self.Iterations = Solver().Iterations

            # inherit methods
            self.sigmoid = Solver().sigmoid
            self.accuracy = Solver().accuracy

        # cost function
        def costFunction(self, Theta, X, Y):
            m = X.shape[0] # number of training examples
            n = X.shape[1] # number of features
            # Theta is a matrix
            Theta = np.asmatrix(Theta).reshape((n, Theta.shape[1])) 
            H = np.dot(X, Theta) # hypothesis
            J = - (np.transpose(Y) * np.log(self.sigmoid(H)) + np.transpose(1 - Y) * np.log(1 - self.sigmoid(H)))

            # introduce regularization (Theta_0 is not regularized)
            if self.Lambda != 0:
                J = J + (self.Lambda / 2.0) * np.sum(np.power(Theta[xrange(1, Theta.shape[0])], 2))

            return J * (1.0 / m)

        # gradient 
        def gradient(self, Theta, X, Y):
            m = X.shape[0] # number of training examples
            n = X.shape[1] # number of features
            # Theta is a matrix
            Theta = np.asmatrix(Theta).reshape((n, Theta.shape[1]))
            H = np.dot(X, Theta) # hypothesis
            grad = np.transpose(X) * (self.sigmoid(H) - Y) + self.Lambda * Theta
            grad[0] = grad[0] - self.Lambda * Theta[0] # grad_0 is not regularized
            grad = grad * (1.0 / m)

            return grad

        # train the Logistic Classifier
        def trainLogisticClassifier(self, Theta, X, Y, Algorithm):
            print 'Training...'
            # lbfgs algorithm
            if Algorithm == 'lbfgs':
                alg = 'L-BFGS'
                # calculate weights
                Theta = (opt.fmin_l_bfgs_b(self.costFunction, Theta, fprime = gradient, args = (X, Y), maxiter = self.Iterations))[0]

            # gradient descent
            elif Algorithm == 'gd':
                alg = 'Gradient Decsent'
                m = Y.shape[0] # number of training examples
                J = mtl.zeros((self.Iterations, 1))  # cost functions
                if self.Alpha != 0:
                    for it in xrange(self.Iterations):  
                        J[it], grad = self.costAndGradient(Theta, X, Y) # cost function and gradient update
                        # calculate weights          
                        Theta = Theta - self.Alpha * grad

                else: print('Please specify a value for Alpha that is different from 0.')

                # monitor process: plot cost functions against iterations
                plt.plot(J, label = r'$\alpha$ = ' + str(self.Alpha))
                plt.xlabel('Iterations')
                plt.title(r'Cost Function - $J(\theta)$')
                plt.legend()
                plt.show()

            else:
                print 'The algorithm selected for optimisation was not identified: ' + Algorithm
                print 'Use "lbfgs" or "gd" instead.'

            # reshape Theta
            Theta = Theta.reshape((Theta.shape[0], Theta.shape[1]))
            print 'Logistic Classifier trained!'
            # show accuracy on training set
            print 'Accuracy on the training set using ' + alg + ': %.2f %%' %(self.accuracy(self.predict(Theta, X), Y))

            return Theta

        # output prediction
        def predict(self, Theta, X):
            m = X.shape[0] # number of training examples
            n = self.output_size # number of labels
            p = mtl.zeros((m, n)) # predictions
            H = np.dot(X, Theta) # hypothesis
            for i in xrange(m):
                for j in xrange(n):
                    if n > 1: # if labels are > 1, calculate the highest probability amongst them
                        if self.sigmoid(H[i, j]) == np.argmax(self.sigmoid(H[i, :])):
                            p[i, j] = 1.0
                    else: # if only 1 label, return 1 if probability is > 0.5
                        if self.sigmoid(H[i, j]) >= 0.5:
                            p[i, j] = 1.0
                    
            return p

        # model testing
        def testModel(self, Theta, X_test, Y_test):

            print 'Accuracy on test set using Logistic Regression: %.2f %%' %(self.accuracy(self.predict(Theta, X_test), Y_test))
