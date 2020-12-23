"""
A class for GP-based path regression
"""
import numpy as np
import math
from gp_code.regression import *
from gp_code.likelihood import nearestPD
from gp_code.sampling import *
from utils.manip_trajectories import goal_center_and_size
from utils.linalg import positive_definite

class path1D_regression:
    # Constructor
    def __init__(self, kernel, sigmaNoise=7.5):
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
        # Regularization factor
        self.epsilon         = 0.5
        self.kernel          = kernel
        self.sigmaNoise      = sigmaNoise

    # Update observations for the Gaussian process (matrix K)
    def updateObservations(self,observedX,observedL,finalX,finalL,finalVar,predictedL):
        # TODO: K could be simply updated instead of being redefined all the time
        n                    = len(observedX)
        self.observedX       = np.zeros((n+1,1))
        self.observedL       = np.zeros((n+1,1))
        self.predictedL      = predictedL
        self.K               = np.zeros((n+1,n+1))
        self.observedL[:-1,0]= observedL
        self.observedX[:-1,0]= observedX
        self.observedL[-1,0] = finalL
        self.observedX[-1,0] = finalX
        # Fill in K, first elements (nxn)
        self.K       = self.kernel(self.observedL[:,0],self.observedL[:,0])
        # Define the variance associated to the last point (varies with the area)
        self.K[n][n]+= finalVar
        # TODO: set the noise parameter somewhere else
        # Heavy
        self.K_1     = inv(self.K+self.sigmaNoise*np.eye(self.K.shape[0]))
        # Center the data in case we use the linear prior
        if self.kernel.linearPrior!=False:
            self.observedX -= (self.kernel.meanSlope*self.observedL+self.kernel.meanConstant)
        # For usage in prediction
        nnew         = len(predictedL)
        self.deltak  = np.zeros((nnew,1))
        self.deltaK  = np.zeros((n+1,1))
        # TODO: vectorize too
        # Fill in deltakx
        for j in range(nnew):
            self.deltak[j][0] = self.kernel.dkdy(self.observedL[n],predictedL[j])
        # Fill in deltaKx
        for j in range(n+1):
            self.deltaK[j][0] = self.kernel.dkdy(self.observedL[n],self.observedL[j])

    # Deduce the mean for the filtered distribution for the observations
    # f = K (K + s I)^-1
    def filterObservations(self):
        f = self.K.dot(self.K_1.dot(self.observedX))
        if self.kernel.linearPrior!=False:
            f +=self.observedL*self.kernel.meanSlope+self.kernel.meanConstant
        return f

    # The main regression function: perform regression for a vector of values
    # lnew, that has been computed in update
    def prediction_to_finish_point(self):
        # No prediction to do
        if self.predictedL.shape[0]==0:
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
        self.k = self.kernel(self.observedL[:,0],self.predictedL[:,0])
        # Fill in C
        # Note here that we do not add the noise term here (we want to recover the unperturbed data)
        # As Eq. 2.22 in Rasmussen
        self.C = self.kernel(self.predictedL[:,0],self.predictedL[:,0])
        # Predictive mean
        self.K_1o       = self.K_1.dot(self.observedX)
        self.predictedX = self.k.transpose().dot(self.K_1o)
        if self.kernel.linearPrior!=False:
            self.predictedX += (self.predictedL*self.kernel.meanSlope+self.kernel.meanConstant)
        # Estimate the variance in x
        self.ktK_1 = self.k.transpose().dot(self.K_1)
        kK_1kt     = self.ktK_1.dot(self.k)
        self.varX  = self.C - kK_1kt
        # Regularization to avoid singular matrices
        self.varX += self.epsilon*np.eye(self.varX.shape[0])
        # Cholesky on varX
        # TODO: use it for the inverse?
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
    def generate_random_variation(self):
        nPredictions = len(self.predictedL)
        sX           = np.random.normal(size=(nPredictions,1))
        if self.sqRootVar.shape[0]>0:
            return self.sqRootVar.dot(sX)
        else:
            return np.zeros(nPredictions,1)
