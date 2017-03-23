# Binary-Logistic-Regression

Module that contains functions to perform Logistic Regression for Binary Classification Machine Learning problems.

Language: Python 2.7

Additional modules: numpy, matplotlib

FUNCTIONS:

1. Sigmoid, as the probability for the event to occur. If the argument overcomes 100 in absolute value, then 100 is used (no change in result)
    
2. binCostFunction, to monitor the cost function:

    a. Theta: parameters vector
    
    b. X, Y: variables and outputs vectors
    
    c. Lambda: regularization parameter
    
3. gradientDescent, to optimise Theta:

    a. Theta, X, Y, Lambda: as in binCostFunction
    
    b. Alpha: learning rate
    
    c. Iterations: number of iterations
    
    d. Plot: boolean variable. If True, the cost function is plotted against the number of iterations, to monitor convergence
        
4. prediction, to predict the output basing on the model obtained after learning. Returns 1 if the hypothesis model results in a value >= 0.5, 0 otherwise:

    a. Theta, X: as in binCostFunction
    
    b. Alg: algorithm used to calculate Theta ('normEq' or 'gradDescent')
