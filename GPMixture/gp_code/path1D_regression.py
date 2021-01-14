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
        # Observation vectors
        self.observedX       = None
        self.observedL       = None
        # Maximal number of observations used to update the GP
        self.observedNMax    = 15
        # Prediction vectors
        self.predictedX      = None
        self.predictedL      = None
        self.K               = None
        self.Kp_1            = None
        self.k               = None
        self.C               = None
        self.sqRootVar       = np.empty((0, 0))
        # Regularization factor
        self.epsilon         = 0.5
        self.kernel          = kernel
        self.sigmaNoise      = sigmaNoise

    # Method to select a maximal number of self.observedNMax filterObservations organized in a logrithmic scale
    def selectObservations(self,observedL,observedX):
        n                    = len(observedX)
        # Number of data we will use
        nm                   = min(n,self.observedNMax)
        idx                  = np.flip(n-np.logspace(0,np.log10(n), num=nm))
        idx                  = idx.astype(int)
        # Search for doubles
        doubles = True
        while doubles:
            doubles = False
            for i in range(nm-2,0,-1):
                # Found a double
                if idx[i]==idx[i+1]:
                    doubles = True
                    for j in range(i,-1,-1):
                        # Could be a place for i
                        if idx[j]-idx[j+1]<-1:
                            tmp   = idx[j+1]-1
                            # Move the data to the right
                            for k in range(i,j,-1):
                                idx[k]=idx[k-1]
                            idx[j+1]= tmp
                            break
                    break
        return observedL[idx],observedX[idx]

    # Update observations for the Gaussian process (matrix K)
    def updateObservations(self,observedX,observedL,finalX,finalL,finalVar,predictedL):
        # Number of "real" observations (we add one: the final point)
        n                    = len(observedX)
        nm                   = min(n,self.observedNMax)
        # Observation vectors: X and L
        self.observedX       = np.zeros((nm+1,1))
        self.observedL       = np.zeros((nm+1,1))
        # Values of arc length L at which we predict X
        self.predictedL      = predictedL
        # Covariance matrix
        self.K               = np.zeros((nm+1,nm+1))
        # Set the observations
        self.observedL[:-1,0], self.observedX[:-1,0] = self.selectObservations(observedL,observedX)
        self.observedL[-1,0] = finalL
        self.observedX[-1,0] = finalX
        # Center the data in case we use the linear prior
        if self.kernel.linearPrior!=False:
            self.observedX -= (self.kernel.meanSlope*self.observedL+self.kernel.meanConstant)
        # Fill in K, first elements (nxn)
        self.K       = self.kernel(self.observedL[:,0],self.observedL[:,0])
        # Add the variance associated to the last point (varies with the area)
        self.K[nm][nm]+= finalVar
        # Heavy
        self.Kp_1    = inv(self.K+self.sigmaNoise*np.eye(self.K.shape[0]))
        self.Kp_1o   = self.Kp_1.dot(self.observedX)
        # For usage in prediction
        nnew         = len(predictedL)
        self.deltak  = np.zeros((nnew,1))
        self.deltaK  = np.zeros((nm+1,1))
        # Fill in deltakx
        self.deltak  = self.kernel.dkdy(self.observedL[nm],predictedL)
        # Fill in deltaKx
        self.deltaK = self.kernel.dkdy(self.observedL[nm],self.observedL)

    # Deduce the mean for the filtered distribution for the observations
    # f = K (K + s I)^-1
    def filterObservations(self):
        f = self.K.dot(self.Kp_1.dot(self.observedX))
        if self.kernel.linearPrior!=False:
            f +=self.observedL*self.kernel.meanSlope+self.kernel.meanConstant
        return f

    # The main regression function: perform regression for a vector of values
    # lnew, that has been computed in update
    def prediction_to_finish_point(self,compute_sqRoot=False):
        # No prediction to do
        if self.predictedL.shape[0]==0:
            return None, None, None
        # Fill in k
        self.k = self.kernel(self.observedL[:,0],self.predictedL[:,0])
        # Fill in C
        # Note here that we **do not add the noise term** here (we want to recover the unperturbed data)
        # As Eq. 2.22 in Rasmussen
        self.C = self.kernel(self.predictedL[:,0],self.predictedL[:,0])
        # Predictive mean
        self.predictedX = self.k.transpose().dot(self.Kp_1o)
        if self.kernel.linearPrior!=False:
            self.predictedX += (self.predictedL*self.kernel.meanSlope+self.kernel.meanConstant)
        # Estimate the variance in x
        self.ktKp_1= self.k.transpose().dot(self.Kp_1)
        kK_1kt     = self.ktKp_1.dot(self.k)
        self.varX  = self.C - kK_1kt
        # Regularization to avoid singular matrices
        self.varX += self.epsilon*np.eye(self.varX.shape[0])
        # Cholesky on varX: done only if the compute_sqRoot flag is true
        if compute_sqRoot and positive_definite(self.varX):
            self.sqRootVar     = cholesky(self.varX,lower=True)
        return self.predictedX, self.varX

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
        newx = self.predictedX + self.ktKp_1.dot(deltaX)
        # First order term #2: variation in kX
        newx+= self.Kp_1o[-1][0]*deltal*self.deltak
        # First order term #3: variation in Kx_1
        #  x= k^T (K+DK)^{-1}x
        #  x= k^T ((I+DK.K^{-1})K)^{-1}x
        #  x= k^T K^{-1}(I+DK.K^{-1})^{-1}x
        # dx=-k^T K^{-1}.DK.K^{-1}x
        newx-= deltal*self.Kp_1o[-1][0]*self.ktKp_1.dot(self.deltaK)
        newx-= deltal*self.ktKp_1[0][-1]*self.deltaK.transpose().dot(self.Kp_1o)
        return self.predictedL, newx, self.varX

    # Generate a random variation to the mean
    def generate_random_variation(self):
        nPredictions = len(self.predictedL)
        sX           = np.random.normal(size=(nPredictions,1))
        if self.sqRootVar.shape[0]>0:
            return self.sqRootVar.dot(sX)
        else:
            return np.zeros(nPredictions,1)
