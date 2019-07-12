"""
A class for handling a single GP in trajectory prediction
"""
import numpy as np
import math
from regression import *
from evaluation import *
from statistics import*
from sampling import *
from gpRegressor import *

# Class for performing path regression with a single Gaussian Process
class singleGP:

    # Constructor
    def __init__(self, startG, endG, stepUnit, goalsData):
        self.goalsData       = goalsData
        self.nPoints         = 5
        self.stepUnit        = stepUnit
        self.startG          = startG
        self.endG            = endG
        self.predictedMeans  = None
        self.predictedVars   = None
        self.likelihood      = 0.0
        # The basic element here is this object, that will do the regression work
        self.gpPathRegressor = gpRegressor(self.goalsData.kernelsX[self.startG][self.endG], self.goalsData.kernelsY[self.startG][self.endG],goalsData.units[self.startG][self.endG],stepUnit,self.goalsData.linearPriorsX[self.startG][self.endG], self.goalsData.linearPriorsY[self.startG][self.endG])

    # Update observations and compute likelihood based on observations
    def update(self,observedX,observedY,observedL):
        goalCenter = middle_of_area(self.goalsData.areas[self.endG])
        # Update observations and re-compute the kernel matrices
        self.gpPathRegressor.updateObservations(observedX,observedY,observedL,goalCenter)
        # Compute the model likelihood
        return self.gpPathRegressor.computeLikelihood(observedX,observedY,observedL,self.startG,self.endG,self.nPoints,self.goalsData)

    # Performs prediction
    def predict(self):
        # Uses the already computed matrices to apply regression over missing data
        predictedX, predictedY, predictedL, varX, varY = self.gpPathRegressor.prediction_to_finish_point()
        self.predictedMeans = np.column_stack((predictedX, predictedY, predictedL))
        self.predictedVars  = [varX, varY]
        return self.predictedMeans,self.predictedVars

    # Generate a sample from the current Gaussian predictive distribution
    def sample(self):
        # Sample end point around the sampled goal
        finishX, finishY, axis = uniform_sampling_1D(1, self.goalsData.areas[self.endG], self.goalsData.areasAxis[self.endG])

        # Use a pertubation approach to get the sample
        goalCenter = middle_of_area(self.goalsData.areas[self.endG])
        deltaX = finishX[0]-goalCenter[0]
        deltaY = finishY[0]-goalCenter[1]
        return self.gpPathRegressor.sample_with_perturbed_finish_point(deltaX,deltaY)

    # Generate samples from the predictive distribution
    def generate_samples(self,nSamples):
        vecX, vecY = [], []
        for k in range(nSamples):
            x, y = self.sample()
            vecX.append(x)
            vecY.append(y)
        return vecX,vecY
