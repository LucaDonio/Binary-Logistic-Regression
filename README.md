# Machine Learning Library

Module that contains functions to perform Machine Learning algorithms. 
This is an ongoing project, so the file will be updated with new functions as soon as they will be implemented.

Language: Python 2.7

Additional modules: numpy, matplotlib

FUNCTIONS:

1. Sigmoid, as the probability for the event to occur
    
2. binCostFunction, to monitor the cost function:

    a. Theta: parameters vector
    
    b. X, Y: variables and outputs vectors
    
    c. Lambda: regularization parameter
    
3. gradientDescent, to optimise Theta:

    a. Theta, X, Y, Lambda: as in binCostFunction
    
    b. Alpha: learning rate
    
    c. Iterations: number of iterations
    
    d. Monitor: boolean variable. If True, the cost function is plotted against the number of iterations to monitor convergence, and the accuracy on the training set is displayed
        
4. prediction, to predict the output basing on the model obtained after learning. Returns 1 if the hypothesis model results in a value >= 0.5, 0 otherwise:

    a. Theta, X: as in binCostFunction
  
NOTE: make sure to include the bias column in matrix X, before passing the latter to any function.
