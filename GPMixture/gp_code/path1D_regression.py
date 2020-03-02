"""
A class for GP-based path regression
"""
import numpy as np
import math
from gp_code.regression import *
from gp_code.likelihood import nearestPD
from gp_code.sampling import *
from utils.manip_trajectories import goal_center_and_size
from utils.manip_trajectories import euclidean_distance, positive_definite

class path1D_regression:
    # Constructor
    def __init__(self, kernel,linearPrior=None):
        # Observations
        self.observedX       = None
        self.observedL       = None
        self.predictedX      = None
        self.predictedL      = None
        self.K               = None
        self.K_1             = None
        self.k               = None
        self.C               = None
        self.sqRootVar       = np.empty((0, 0))
        self.epsilon         = 0.5
        self.kernel          = kernel
        self.linearPrior     = linearPrior

    # Update observations for the Gaussian process (matrix K)
    def updateObservations(self,observedX,observedL,finalX,finalL,finalVar,predictedL):
        # TODO: K could be simply updated instead of being redefined all the time
        n                    = len(observedX)
        self.observedX       = np.zeros((n+1,1))
        self.observedL       = np.zeros((n+1,1))
        self.predictedL      = predictedL
        self.K               = np.zeros((n+1,n+1))
        # Fill in K, first elements (nxn)
        for i in range(n):
            self.K[i][i] = self.kernel(observedL[i],observedL[i])
            for j in range(i):
                self.K[i][j] = self.kernel(observedL[i],observedL[j])
                self.K[j][i] = self.K[i][j]
        # Last row/column
        for i in range(n):
            self.K[i][n] = self.kernel(observedL[i],finalL)
            self.K[n][i] = self.K[i][n]
        self.K[n][n] = self.kernel(finalL,finalL)

        # Define the variance associated to the last point (varies with the area)
        self.K[n][n] += finalVar
        # Heavy
        self.K_1 = inv(self.K)
        for i in range(n):
            self.updateObserved(i,observedX[i],observedL[i])
        self.updateObserved(n,finalX,finalL)
        # For usage in prediction
        nnew         = len(predictedL)
        self.deltak  = np.zeros((nnew,1))
        self.deltaK  = np.zeros((n+1,1))
        # Fill in deltakx
        for j in range(nnew):
            self.deltak[j][0] = self.kernel.dkdy(self.observedL[n],predictedL[j])
        # Fill in deltaKx
        for j in range(n+1):
            self.deltaK[j][0] = self.kernel.dkdy(self.observedL[n],self.observedL[j])

    # Update single observation i
    def updateObserved(self,i,x,l):
        # Center the data in case we use the linear prior
        if self.linearPrior==None:
            self.observedX[i][0] = x
            self.observedL[i][0] = l
        else:
            self.observedX[i][0] = x - linear_mean(l, self.linearPrior[0])
            self.observedL[i][0] = l

    # The main regression function: perform regression for a vector of values
    # lnew, that has been computed in update
    def prediction_to_finish_point(self):
        if self.predictedL==None:
            return None, None, None
        # Number of observed data
        n    = self.observedX.shape[0]
        # Number of predicted data
        nnew = len(self.predictedL)
        if nnew == 0:
            return None, None, None
        # Compute k (nxnnew), C (nnewxnnew)
        self.k  = np.zeros((n,nnew))
        self.C  = np.zeros((nnew,nnew))

        # Fill in k
        for i in range(n):
            for j in range(nnew):
                self.k[i][j] = self.kernel(self.observedL[i],self.predictedL[j],False)
        # Fill in C
        for i in range(nnew):
            for j in range(nnew):
                self.C[i][j] = self.kernel(self.predictedL[i],self.predictedL[j],False)

        # Predictive mean
        self.K_1o= self.K_1.dot(self.observedX)
        self.predictedX = self.k.transpose().dot(self.K_1o)
        if self.linearPrior!=None:
            for j in range(nnew):
                self.predictedX[j] += linear_mean(self.predictedL[j],self.linearPrior[0])
        # Estimate the variance in x
        self.ktK_1 = self.k.transpose().dot(self.K_1)
        kK_1kt     = self.ktK_1.dot(self.k)
        self.varX  = self.C - kK_1kt
        # Regularization to avoid singular matrices
        self.varX += self.epsilon*np.eye(self.varX.shape[0])
        # Cholesky on varX
        if positive_definite(self.varX):
            try:
                self.sqRootVar = np.linalg.cholesky(self.varX)
            except np.linalg.LinAlgError:
                    self.varX = nearestPD(self.varX)
            self.sqRootVar     = cholesky(self.varX,lower=True)
        return self.predictedL, self.predictedX, self.varX

    # Prediction as a perturbation of the "normal" prediction done to the center of an area
    def prediction_to_perturbed_finish_point(self,deltal,deltax):
        n            = len(self.observedX)
        npredicted   = len(self.predictedL)
        if npredicted == 0:
            return None, None, None
        # Express the displacement wrt the nominal ending point, as a nx1 vector
        deltaX       = np.zeros((n,1))
        deltaX[n-1,0]= deltax
        # In this approximation, only the predictive mean is adapted (will be used for sampling)
        # First order term #1: variation in observedX
        newx = self.predictedX + self.ktK_1.dot(deltaX)
        # First order term #2: variation in kX
        newx+= self.K_1o[-1][0]*deltal*self.deltak
        # First order term #3: variation in Kx_1
        #  x= k^T (K+DK)^{-1}x
        #  x= k^T ((I+DK.K^{-1})K)^{-1}x
        #  x= k^T K^{-1}(I+DK.K^{-1})^{-1}x
        # dx=-k^T K^{-1}.DK.K^{-1}x
        newx-= deltal*self.K_1o[-1][0]*self.ktK_1.dot(self.deltaK)
        newx-= deltal*self.ktK_1[0][-1]*self.deltaK.transpose().dot(self.K_1o)
        return self.predictedL, newx, self.varX

    # Generate a random variation to the mean
    def generate_variation(self):
        nPredictions = len(self.predictedL)
        sX           = np.random.normal(size=(nPredictions,1))
        if self.sqRootVar.shape[0]>0:
            return self.sqRootVar.dot(sX)
        else:
            return np.zeros(nPredictions,1)
