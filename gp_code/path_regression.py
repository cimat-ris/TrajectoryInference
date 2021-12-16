"""
A class for GP-based path regression
"""
import numpy as np
import math
from gp_code.regression import *
from gp_code.sampling import *
from gp_code.path1D_regression import path1D_regression
from gp_code.onedim_regressionT import onedim_regressionT
from utils.manip_trajectories import goal_center_and_size
from utils.stats_trajectories import euclidean_distance
from gp_code.likelihood import ADE
from scipy.optimize import bisect
import time, logging

class path_regression:
    # Constructor
    def __init__(self, kernelX, kernelY, sigmaNoise, unit, finalArea, prior=0.0, timeTransitionData=None):
        self.regression_x    = path1D_regression(kernelX,sigmaNoise)
        self.regression_y    = path1D_regression(kernelY,sigmaNoise)
        self.predictedL      = None
        self.distUnit        = unit
        self.stepUnit        = 1.0
        self.finalArea       = finalArea
        self.finalAreaCenter, self.finalAreaSize = goal_center_and_size(finalArea[1:])
        self.finalAreaAxis   = finalArea[0]
        self.prior           = prior

    # Update observations x,y,l for the Gaussian process (matrix K)
    def update_observations(self,observations,consecutiveObservations=True):
        # Last really observed point
        lastObs = observations[-1]
        if lastObs[2] > 1:
            self.stepUnit = float(len(observations[:,2:3])/lastObs[2])
        # Determine the set of arclengths (predictedL) to predict
        self.predictedL, finalL, self.dist = self.prediction_set_arclength(lastObs,self.finalAreaCenter)

        # Define the variance associated to the last point (varies with the area)
        if self.finalAreaAxis==0:
            s              = self.finalAreaSize[0]
        else:
            s              = self.finalAreaSize[1]
        # Update observations of each process (x,y)
        self.regression_x.update_observations(observations[:,0:1],observations[:,2:3],self.finalAreaCenter[0],finalL,(1.0-self.finalAreaAxis)*s*s*math.exp(-self.dist/s),self.predictedL,consecutiveObservations)
        self.regression_y.update_observations(observations[:,1:2],observations[:,2:3],self.finalAreaCenter[1],finalL,    (self.finalAreaAxis)*s*s*math.exp(-self.dist/s),self.predictedL,consecutiveObservations)


    def prediction_set_arclength(self, lastObs, finishPoint):
        # Coordinates of the last observed point
        x, y, l = lastObs[0], lastObs[1], lastObs[2]
        # Euclidean distance between the last observed point and the finish point
        euclideanDist = euclidean_distance([x,y], finishPoint)
        # Rough estimate of the remaining arc length
        distToGoal    = euclideanDist*self.distUnit
        size          = int(distToGoal*self.stepUnit)
        predset = np.zeros((size,1))
        if size > 0:
            step = distToGoal/float(size)
            for i in range(1,size+1):
                predset[i-1,0] = l + i*step
        return predset, l + distToGoal, distToGoal

    # Prediction set for Trautman's approach
    def prediction_set_time(self, lastObs, finishPoint, elapsedTime, timeStep):
        # Time of the last observed point
        x, y, t = lastObs[0], lastObs[1], lastObs[2]
        distToGoal = euclidean_distance([x,y], finishPoint)
        # TODO: I think we should first do here a fully deterministic model (conditioned on the mean transition time)
        # Sample a duration
        transitionTime = int(np.random.normal(self.timeTransitionMean, self.timeTransitionStd) )

        # Remaining time
        remainingTime = transitionTime - elapsedTime
        #!Problem: timeTransitionMean <= elapsedTime
        if remainingTime <= 0:
            return np.zeros((0,1)), 0, 0
        size = int(remainingTime/timeStep)
        predset = np.zeros((size,1))
        if size > 0:
            for i in range(1,size+1):
                predset[i-1,0] = t + i*timeStep
            if predset[-1,0] < t + remainingTime:
                predset[-1,0] = t + remainingTime
        return predset, t + remainingTime, distToGoal

    # Filter initial observations
    def filter_observations(self):
        filteredx = self.regression_x.filter_observations()
        filteredy = self.regression_y.filter_observations()
        return np.concatenate([filteredx,filteredy],axis=1)

    # Compute the log likelihood independently on x and y, the sum them.
    # Each call also returns the predicted piece of trajectory
    def loglikelihood_from_partial_path(self):
        llx, predx  = self.regression_x.loglikelihood_from_partial_path()
        lly, predy =  self.regression_y.loglikelihood_from_partial_path()
        if predx is None or predy is None:
            pred = None
        else:
            pred = np.concatenate([predx,predy],axis=1)
        return llx+lly,pred

    # Compute the likelihood
    def compute_likelihood(self):
        ll,preds = self.loglikelihood_from_partial_path()
        self.likelihood = self.prior*np.exp(ll)
        return self.likelihood,preds

    # The main path regression function: perform regression for a
    # vector of values of future L, that has been computed in update
    def predict_path_to_finish_point(self,compute_sqRoot=False):
        if self.predictedL.shape[0] == 0:
            return None, None
        predx, varx = self.regression_x.predict_to_finish_point(compute_sqRoot=compute_sqRoot)
        predy, vary = self.regression_y.predict_to_finish_point(compute_sqRoot=compute_sqRoot)
        return np.concatenate([predx, predy, self.predictedL],axis=1),np.stack([varx,vary],axis=0)

    # Generate a sample from perturbations
    def sample_path_with_perturbation(self,deltaX,deltaY,efficient=True):
        # A first order approximation of the new final value of L
        deltaL       = deltaX*(self.finalAreaCenter[0]-self.regression_x.observedX[-2])/self.dist + deltaY*(self.finalAreaCenter[1]-self.regression_y.observedX[-2])/self.dist
        if efficient:
            # Given a perturbation of the final point, determine the new characteristics of the GP
            predictedL,predictedX,__=self.regression_x.predict_to_perturbed_finish_point(deltaL,deltaX)
            predictedL,predictedY,__=self.regression_y.predict_to_perturbed_finish_point(deltaL,deltaY)
        else:
            predictedL,predictedX,__=self.regression_x.predict_to_perturbed_finish_point_slow(deltaL,deltaX)
            predictedL,predictedY,__=self.regression_y.predict_to_perturbed_finish_point_slow(deltaL,deltaY)

        if predictedX is None or predictedY is None:
            return None#,None,None
        # Generate a sample from this Gaussian distribution
        nx=self.regression_x.generate_random_variation()
        ny=self.regression_y.generate_random_variation()
        return np.concatenate([predictedX+nx,predictedY+ny,predictedL],axis=1)

    # Generate a sample from the predictive distribution with a perturbed finish point
    def sample_path_with_perturbed_finish_point(self,efficient=True):
        # Sample end point around the sampled goal
        size = self.finalAreaSize[self.finalArea[0]]
        finishX, finishY, axis = uniform_sampling_1D_around_point(1, self.finalAreaCenter,size, self.finalAreaAxis)
        # Use a pertubation approach to get the sample
        deltaX = finishX[0]-self.finalAreaCenter[0]
        deltaY = finishY[0]-self.finalAreaCenter[1]
        return self.sample_path_with_perturbation(deltaX,deltaY,efficient)
