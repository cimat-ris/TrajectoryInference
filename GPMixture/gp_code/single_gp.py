"""
A class for handling a single GP in trajectory prediction
"""
import numpy as np
import math
from gp_code.path_regression import *

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
        self.gpPathRegressor = path_regression(self.goalsData.kernelsX[self.startG][self.endG], self.goalsData.kernelsY[self.startG][self.endG],self.goalsData.units[self.startG][self.endG],self.goalsData.stepUnit,self.goalsData.areas_coordinates[self.endG],self.goalsData.areas_axis[self.endG],self.goalsData.priorTransitions[self.startG][self.endG])

    # Update observations and compute likelihood based on observations
    def update(self,observedX,observedY,observedL):
        # Update observations and re-compute the kernel matrices
        self.gpPathRegressor.updateObservations(observedX,observedY,observedL)
        # Compute the model likelihood
        return self.gpPathRegressor.computeLikelihood(observedX,observedY,observedL,self.nPoints)

    # Performs prediction
    def predict(self,compute_sqRoot=False):
        # Uses the already computed matrices to apply regression over missing data
        predictedX, predictedY, predictedL, varX, varY = self.gpPathRegressor.prediction_to_finish_point(compute_sqRoot=compute_sqRoot)
        self.predictedMeans = np.column_stack((predictedX, predictedY, predictedL))
        self.predictedVars  = np.stack([varX, varY],axis=0)
        return self.predictedMeans,self.predictedVars

    # Get a filtered version of the initial observations
    def filter(self):
        return self.gpPathRegressor.filterObservations()

    # Generate a sample from the current Gaussian predictive distribution
    def sample(self):
        return self.gpPathRegressor.sample_with_perturbed_finish_point()

    # Generate samples from the predictive distribution
    def generate_samples(self,nSamples):
        vecX, vecY = [], []
        for k in range(nSamples):
            x, y, __ = self.sample()
            vecX.append(x)
            vecY.append(y)
        return vecX,vecY
