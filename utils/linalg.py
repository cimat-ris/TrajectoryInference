import numpy as np
import math
import logging
from termcolor import colored

# Check a matrix for: negative eigenvalues, asymmetry and negative diagonal values
def positive_definite(M,epsilon = 0.000001,verbose=False):
    # Symmetrization
    Mt = np.transpose(M)
    M = (M + Mt)/2
    eigenvalues = np.linalg.eigvals(M)
    for i in range(len(eigenvalues)):
        if eigenvalues[i] <= epsilon:
            if verbose:
                logging.error("Negative eigenvalues")
            return 0
    for i in range(M.shape[0]):
        if M[i][i] < 0:
            if verbose:
                logging.error("Negative value in diagonal")
            return 0
    return 1
