import numpy as np
import math
from termcolor import colored

# Check a matrix for: neagtive eigenvalues, asymmetry and negative diagonal values
def positive_definite(M):
    eigenvalues = np.linalg.eigvals(M)
    for i in range(len(eigenvalues)):
        if eigenvalues[i] <= 0:
            print(colored("[ERR] Negative eigenvalues ",'red'))
            return 0
    Mt = np.transpose(M)
    M = (M + Mt)/2
    for i in range(M.shape[0]):
        if M[i][i] < 0:
            print(colored("[ERR] Negative value in diagonal",'red'))
            return 0
    return 1
