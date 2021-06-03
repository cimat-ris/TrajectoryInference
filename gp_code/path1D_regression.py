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
    def __init__(self, kernel, sigmaNoise):
        # Observation vectors
        self.observedX       = None
        self.observedL       = None
        self.observedL_2m    = None
        self.observedL_3m    = None
        # Prediction vectors
        self.predictedX      = None
        self.predictedL      = None
        # Covariance matrix
        self.K               = None
        # Inverse of the previous
        self.Kp_1            = None
        self.Kp_1_2m         = None
        self.Kp_1_3m         = None
        self.Kp_1o           = None
        self.Kp_1o_2m        = None
        self.Kp_1o_3m        = None
        self.k               = None
        self.C               = None
        self.sqRootVar       = np.empty((0, 0))
        # Selection factor
        self.epsilonSel      = 0.33
        # Regularization factor
        self.epsilonReg      = 0.5
        # Kernel
        self.kernel          = kernel
        # Noise
        self.sigmaNoise      = sigmaNoise
        #
        self.m               = 5
        self.n               = 0

    # Method to select observations
    def select_observations(self,observedL,observedX):
        # Number of data we will use
        k       = int(np.log(self.n)/np.log(1+self.epsilonSel))
        l2      = np.array(range(2,k+1))
        idx     = self.n-(np.power(1+self.epsilonSel,l2)).astype(int)
        return observedL[idx],observedX[idx]

    # Update observations for the Gaussian process
    def update_observations(self,observedX,observedL,finalX,finalL,finalVar,predictedL):
        # Number of "real" observations (we add one: the final point)
        self.n               = len(observedX)
        # Keep the solution for future likelihood evaluations
        if self.n%self.m==0:
            self.Kp_1_3m = self.Kp_1_2m
            self.Kp_1_2m = self.Kp_1
            self.Kp_1o_3m= self.Kp_1o_2m
            self.Kp_1o_2m= self.Kp_1o
            self.observedL_3m = self.observedL_2m
            self.observedL_2m = self.observedL
        # Values of arc length L at which we will predict X
        self.predictedL      = predictedL
        # Set the observations
        selectedL, selectedX = self.select_observations(observedL,observedX)
        n_obs                = selectedL.shape[0]
        # Covariance matrix
        self.K               = np.zeros((n_obs+1,n_obs+1))
        # Observation vectors: X and L
        self.observedL       = np.zeros((n_obs+1,1))
        self.observedX       = np.zeros((n_obs+1,1))
        self.observedL[:-1]  = selectedL
        self.observedX[:-1]  = selectedX
        self.observedL[-1,0] = finalL
        self.observedX[-1,0] = finalX
        # Center the data in case we use the linear prior
        if self.kernel.linearPrior!=False:
            self.observedX -= (self.kernel.meanSlope*self.observedL+self.kernel.meanConstant)
        # Fill in K, first elements (nxn)
        self.K       = self.kernel(self.observedL[:,0],self.observedL[:,0])
        # Add the variance associated to the last point (varies with the area)
        self.finalVar        = finalVar
        self.K[n_obs][n_obs]+= finalVar
        self.Kp_1    = inv(self.K+self.sigmaNoise*np.eye(self.K.shape[0]))
        self.Kp_1o   = self.Kp_1.dot(self.observedX)
        # For usage in prediction
        nnew         = len(predictedL)
        self.deltak  = np.zeros((nnew,1))
        self.deltaK  = np.zeros((n_obs+1,1))
        # Fill in deltakx
        self.deltak  = self.kernel.dkdy(self.observedL[n_obs],predictedL)
        # Fill in deltaKx
        self.deltaK = self.kernel.dkdy(self.observedL[n_obs],self.observedL)

    # Deduce the mean for the filtered distribution for the observations
    # f = K (K + s I)^-1
    # i.e. the most probable noise-free trajectory
    def filter_observations(self):
        f = self.K.dot(self.Kp_1.dot(self.observedX))
        if self.kernel.linearPrior!=False:
            f +=self.observedL*self.kernel.meanSlope+self.kernel.meanConstant
        return f

    # Compute the log-likelihood for this coordinates
    def loglikelihood_from_partial_path(self):
        if self.Kp_1_3m is None:
            return 1.0
        # We remove the last element (corresponds to goal)
        mL        = np.max(self.observedL_3m[:-1,0])
        idx       = self.observedL>mL
        predictedL= self.observedL[idx].reshape((-1,1))
        trueX     = self.observedX[idx].reshape((-1,1))
        k_3m      = self.kernel(self.observedL_3m[:,0],predictedL[:,0])
        C_3m      = self.kernel(predictedL[:,0],predictedL[:,0])
        # Predictive mean
        predictedX_3m = k_3m.transpose().dot(self.Kp_1o_3m)
        error         = predictedX_3m-trueX
        # Estimate the variance in the predicted x
        ktKp_1_3m = k_3m.transpose().dot(self.Kp_1_3m)
        varX_3m   = C_3m - ktKp_1_3m.dot(k_3m)
        # Regularization to avoid singular matrices
        varX_3m  += (self.epsilonReg+self.sigmaNoise)*np.eye(varX_3m.shape[0])
        errorSq   = np.divide(np.square(error),np.diagonal(varX_3m).reshape((-1,1)))
        return -errorSq.sum()

    # The main regression function: perform regression for a vector of values
    # lnew, that has been computed in update
    def predict_to_finish_point(self,compute_sqRoot=False):
        # No prediction to do
        if self.predictedL.shape[0]==0:
            return None, None, None
        # Fill in k
        self.k = self.kernel(self.observedL[:,0],self.predictedL[:,0])
        # Fill in C
        # Note here that we **do not add the noise term** here
        # (we want to recover the unperturbed data)
        # As Eq. 2.22 in Rasmussen
        self.C = self.kernel(self.predictedL[:,0],self.predictedL[:,0])
        # Predictive mean
        self.predictedX = self.k.transpose().dot(self.Kp_1o)
        # When using a linear prior, we need to add it again
        if self.kernel.linearPrior!=False:
            self.predictedX += (self.predictedL*self.kernel.meanSlope+self.kernel.meanConstant)
        # Estimate the variance in the predicted x
        self.ktKp_1= self.k.transpose().dot(self.Kp_1)
        self.varX  = self.C - self.ktKp_1.dot(self.k)
        # Regularization to avoid singular matrices
        self.varX += self.epsilonReg*np.eye(self.varX.shape[0])
        # Cholesky on varX: done only if the compute_sqRoot flag is true
        if compute_sqRoot and positive_definite(self.varX):
            self.sqRootVar     = cholesky(self.varX,lower=True)
        return self.predictedX, self.varX

    # Prediction as a perturbation of the "normal" prediction done to the center of an area
    def predict_to_perturbed_finish_point(self,deltal,deltax):
        n            = len(self.observedX)
        npredicted   = len(self.predictedL)
        if npredicted == 0:
            return None, None, None
        # Express the displacement wrt the nominal ending point, as a nx1 vector
        deltaX       = np.zeros((n,1))
        deltaX[n-1,0]= deltax
        # TODO: revise terms from the equations in the paper
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
        npredicted = len(self.predictedL)
        sX           = np.random.normal(size=(npredicted,1))
        if self.sqRootVar.shape[0]>0:
            return self.sqRootVar.dot(sX)
        else:
            return np.zeros((npredicted,1))
