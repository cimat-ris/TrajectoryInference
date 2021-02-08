"""
A class for handling a single GP in trajectory prediction
"""
import numpy as np
import math
from gp_code.trajectory_regression import *

# Class for performing path regression with a single Gaussian Process
class singleGP:

    # Constructor
    def __init__(self, startG, endG, goalsData):
        self.goalsData       = goalsData
        self.nPoints         = 5
        self.startG          = startG
        self.endG            = endG
        self.predictedMeans  = None
        self.predictedVars   = None
        # The basic element here is this object, that will do the regression work
        self.gpPathRegressor = trajectory_regression(self.goalsData.kernelsX[self.startG][self.endG], self.goalsData.kernelsY[self.startG][self.endG],self.goalsData.units[self.startG][self.endG],self.goalsData.stepUnit,self.goalsData.areas_coordinates[self.endG],self.goalsData.areas_axis[self.endG],self.goalsData.speedModels[self.startG][self.endG],self.goalsData.priorTransitions[self.startG][self.endG])

    # Update observations and compute likelihood based on observations
    def update(self,observations):
        # Update observations and re-compute the kernel matrices
        self.gpPathRegressor.update_observations(observations)
        # Compute the model likelihood
        return self.gpPathRegressor.compute_likelihood(observations,self.nPoints)

    # Performs prediction
    def predict_path(self,compute_sqRoot=False):
        # Uses the already computed matrices to apply regression over missing data
        predictedX, predictedY, predictedL, varX, varY = self.gpPathRegressor.predict_path_to_finish_point(compute_sqRoot=compute_sqRoot)
        self.predictedMeans = np.column_stack((predictedX, predictedY, predictedL))
        self.predictedVars  = np.stack([varX, varY],axis=0)
        return self.predictedMeans,self.predictedVars

    # Get a filtered version of the initial observations
    def filter(self):
        return self.gpPathRegressor.filter_observations()

    # Generate a sample from the current Gaussian predictive distribution
    def sample_path(self):
        return self.gpPathRegressor.sample_path_with_perturbed_finish_point()

    # Generate samples from the predictive distribution
    def sample_paths(self,nSamples):
        vecX, vecY = [], []
        for k in range(nSamples):
            x, y, __ = self.sample_path()
            vecX.append(x)
            vecY.append(y)
        return vecX,vecY
