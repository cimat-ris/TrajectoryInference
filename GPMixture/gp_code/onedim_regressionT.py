"""
A class for GP-based one-dimensional regression | Trautman
"""
import numpy as np
import math
from gp_code.sampling import *
from utils.linalg import positive_definite

class onedim_regressionT:
    # Constructor
    def __init__(self, kernel, sigmaNoise=7.5):
        # Observation vectors
        self.observedX       = None     #independent variable
        self.observedY       = None     #dependent variable
        # Prediction vectors
        self.predictedX      = None
        self.predictedY      = None
        self.K               = None
        self.Kp_1            = None
        self.k               = None
        self.C               = None
        self.sqRootVar       = np.empty((0, 0))
        # Regularization factor
        self.epsilon         = 0.5
        self.kernel          = kernel
        self.sigmaNoise      = sigmaNoise
        
    #TODO: adapt function to Trautman's paper
    # Update observations for the Gaussian process (matrix K)
    def updateObservations(self,observedY,observedX,finalY,finalX,finalVar,predictedX):
        # Number of "real" observations (we add one: the final point) --> check!
        n                    = len(observedY)
        # Observations (x,y)
        self.observedX       = np.zeros((n+1,1))
        self.observedY       = np.zeros((n+1,1))
        # Values of X at which we predict Y
        self.predictedX      = predictedX
        # Covariance matrix
        self.K               = np.zeros((n+1,n+1))
        # Set the observations
        self.observedX[:-1,0], self.observedY[:-1,0] = observedX, observedY
        self.observedX[-1,0] = finalX
        self.observedY[-1,0] = finalY
        
        # Fill in K, first elements (nxn)
        self.K       = self.kernel(self.observedX[:,0],self.observedX[:,0])
        # Add the variance associated to the last point (varies with the area)
        self.finalVar  = finalVar
        self.K[n][n]   += finalVar
        # Heavy
        self.Kp_1    = inv(self.K+self.sigmaNoise*np.eye(self.K.shape[0]))
        self.Kp_1o   = self.Kp_1.dot(self.observedY)
        # For usage in prediction
        nnew         = len(predictedX)
        self.deltak  = np.zeros((nnew,1))
        self.deltaK  = np.zeros((n+1,1))
        # Fill in deltakx
        self.deltak  = self.kernel.dkdy(self.observedX[n],predictedX)
        # Fill in deltaKx
        self.deltaK = self.kernel.dkdy(self.observedX[n],self.observedX)
