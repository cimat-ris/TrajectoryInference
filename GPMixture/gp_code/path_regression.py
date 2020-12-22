"""
A class for GP-based path regression
"""
import numpy as np
import math
from gp_code.regression import *
from gp_code.path1D_regression import path1D_regression
from gp_code.likelihood import likelihood_from_partial_path
from gp_code.sampling import *
from utils.manip_trajectories import goal_center_and_size
from scipy.optimize import bisect

class path_regression:
    # Constructor
    def __init__(self, kernelX, kernelY, unit, stepUnit, finalArea, finalAreaAxis):
        self.regression_x    = path1D_regression(kernelX)
        self.regression_y    = path1D_regression(kernelY)
        self.predictedL      = None
        self.distUnit        = unit
        self.stepUnit        = stepUnit
        self.finalArea       = finalArea
        self.finalAreaAxis   = finalAreaAxis
        self.finalAreaCenter, self.finalAreaSize = goal_center_and_size(finalArea)

    # Update observations for the Gaussian process (matrix K)
    def updateObservations(self,observedX,observedY,observedL):
        # Last really observed point
        lastObservedPoint = [observedX[-1], observedY[-1], observedL[-1]]
        # Determine the set of arclengths (predictedL) to predict
        self.predictedL, finalL, self.dist = get_prediction_set_arclengths(lastObservedPoint,self.finalAreaCenter,self.distUnit,self.stepUnit)
        # Define the variance associated to the last point (varies with the area)
        if self.finalAreaAxis==0:
            s              = self.finalAreaSize[0]
        elif self.finalAreaAxis==1:
            s              = self.finalAreaSize[1]
        # Update observations of each process
        self.regression_x.updateObservations(observedX,observedL,self.finalAreaCenter[0],finalL,(1.0-self.finalAreaAxis)*s*s*math.exp(-self.dist/s),self.predictedL)
        self.regression_y.updateObservations(observedY,observedL,self.finalAreaCenter[1],finalL,    (self.finalAreaAxis)*s*s*math.exp(-self.dist/s),self.predictedL)

    # Filter initial observations
    def filterObservations(self):
        filteredx = self.regression_x.filterObservations()
        filteredy = self.regression_y.filterObservations()
        return filteredx,filteredy
        
    # Compute the likelihood
    def computeLikelihood(self,observedX,observedY,observedL,startG,finishG,stepsToCompare,goalsData):
        # TODO: remove the goalsData structure
        self.likelihood = goalsData.priorTransitions[startG][finishG]*likelihood_from_partial_path(stepsToCompare,observedX,observedY,observedL,startG,finishG,goalsData)
        return self.likelihood

    # The main regression function: perform regression for a
    # vector of values of L, that has been computed in update
    def prediction_to_finish_point(self):
        pL,pX,vX = self.regression_x.prediction_to_finish_point()
        pL,pY,vY = self.regression_y.prediction_to_finish_point()
        return pX, pY, pL, vX, vY

    # Generate a sample from perturbations
    def sample_with_perturbation(self,deltaX,deltaY):
        # A first order approximation of the new final value of L
        deltaL       = deltaX*(self.finalAreaCenter[0]-self.regression_x.observedX[-2])/self.dist + deltaY*(self.finalAreaCenter[1]-self.regression_y.observedX[-2])/self.dist
        # Given a perturbation of the final point, determine the new characteristics of the GP
        predictedL,predictedX,__=self.regression_x.prediction_to_perturbed_finish_point(deltaL,deltaX)
        predictedL,predictedY,__=self.regression_y.prediction_to_perturbed_finish_point(deltaL,deltaY)
        if predictedX is None or predictedY is None:
            return None,None,None
        # Generate a sample from this Gaussian distribution
        return predictedX+self.regression_x.generate_random_variation(),predictedY+self.regression_y.generate_random_variation(), predictedL

    # Generate a sample from the predictive distribution with a perturbed finish point
    def sample_with_perturbed_finish_point(self):
        # Sample end point around the sampled goal
        size = self.finalAreaSize[self.finalAreaAxis]
        finishX, finishY, axis = uniform_sampling_1D_around_point(1, self.finalAreaCenter,size, self.finalAreaAxis)
        # Use a pertubation approach to get the sample
        deltaX = finishX[0]-self.finalAreaCenter[0]
        deltaY = finishY[0]-self.finalAreaCenter[1]
        return self.sample_with_perturbation(deltaX,deltaY)
