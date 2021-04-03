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

class path_regression:
    # Constructor
    def __init__(self, kernelX, kernelY, sigmaNoise, unit, stepUnit, finalArea, prior=0.0,  mode=None, timeTransitionData=None):
        self.regression_x    = onedim_regressionT(kernelX) if mode == 'Trautman' else path1D_regression(kernelX,sigmaNoise)
        self.regression_y    = onedim_regressionT(kernelY) if mode == 'Trautman' else path1D_regression(kernelY,sigmaNoise)
        self.predictedL      = None
        self.distUnit        = unit
        self.stepUnit        = stepUnit
        self.finalArea       = finalArea
        self.finalAreaCenter, self.finalAreaSize = goal_center_and_size(finalArea[1:])
        self.finalAreaAxis   = finalArea[0]
        self.prior           = prior
        self.mode            = mode
        if mode == 'Trautman':
            self.timeTransitionMean = timeTransitionData[0]
            self.timeTransitionStd  = timeTransitionData[1]

    # Update observations x,y,l for the Gaussian process (matrix K)
    def update_observations(self,observations):
        # Last really observed point
        lastObs = observations[-1]
        if self.mode == 'Trautman':
            elapsedTime = observations[:,2:3][-1][0] - observations[:,2:3][0][0]
            timeStep    = observations[:,2:3][1][0] - observations[:,2:3][0][0]
            self.predictedL, finalL, self.dist = self.prediction_set_time(lastObs, self.finalAreaCenter, elapsedTime, timeStep)
            if self.dist < 1.0:
                self.dist = 1.0
        else:
            # Determine the set of arclengths (predictedL) to predict
            self.predictedL, finalL, self.dist = self.prediction_set_arclength(lastObs,self.finalAreaCenter)

        # Define the variance associated to the last point (varies with the area)
        if self.finalAreaAxis==0:
            s              = self.finalAreaSize[0]
        else:
            s              = self.finalAreaSize[1]
        # Update observations of each process (x,y)
        self.regression_x.update_observations(observations[:,0:1],observations[:,2:3],self.finalAreaCenter[0],finalL,(1.0-self.finalAreaAxis)*s*s*math.exp(-self.dist/s),self.predictedL)
        self.regression_y.update_observations(observations[:,1:2],observations[:,2:3],self.finalAreaCenter[1],finalL,    (self.finalAreaAxis)*s*s*math.exp(-self.dist/s),self.predictedL)


    def prediction_set_arclength(self, lastObs, finishPoint):
        # Coordinates of the last observed point
        x, y, l = lastObs[0], lastObs[1], lastObs[2]
        print("[INF] Current point ",x,y)
        print("[INF] Last point ",finishPoint)
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

    def loglikelihood_from_partial_path(self):
        px = self.regression_x.loglikelihood_from_partial_path()
        py = self.regression_y.loglikelihood_from_partial_path()
        return px+py

    # Compute the likelihood
    def compute_likelihood(self):
        if self.mode == 'Trautman':
            self.likelihood = self.loglikelihood_from_partial_path()
        else:
            # TODO:
            self.likelihood = self.prior*np.exp(self.loglikelihood_from_partial_path())
        return self.likelihood

    # The main path regression function: perform regression for a
    # vector of values of future L, that has been computed in update
    def predict_path_to_finish_point(self,compute_sqRoot=False):
        if self.predictedL.shape[0] == 0:
            return np.zeros((0,1)), np.zeros((0,1))
        predx, varx = self.regression_x.predict_to_finish_point(compute_sqRoot=compute_sqRoot)
        predy, vary = self.regression_y.predict_to_finish_point(compute_sqRoot=compute_sqRoot)
        return np.concatenate([predx, predy, self.predictedL],axis=1),np.stack([varx,vary],axis=0)

    # Generate a sample from perturbations
    def sample_path_with_perturbation(self,deltaX,deltaY):
        # A first order approximation of the new final value of L
        deltaL       = deltaX*(self.finalAreaCenter[0]-self.regression_x.observedX[-2])/self.dist + deltaY*(self.finalAreaCenter[1]-self.regression_y.observedX[-2])/self.dist
        # Given a perturbation of the final point, determine the new characteristics of the GP
        predictedL,predictedX,__=self.regression_x.predict_to_perturbed_finish_point(deltaL,deltaX)
        predictedL,predictedY,__=self.regression_y.predict_to_perturbed_finish_point(deltaL,deltaY)
        if predictedX is None or predictedY is None:
            return None,None,None
        # Generate a sample from this Gaussian distribution
        return np.concatenate([predictedX+self.regression_x.generate_random_variation(),predictedY+self.regression_y.generate_random_variation(), predictedL],axis=1)

    # Generate a sample from the predictive distribution with a perturbed finish point
    def sample_path_with_perturbed_finish_point(self):
        # Sample end point around the sampled goal

        size = self.finalAreaSize[self.finalArea[0]]
        finishX, finishY, axis = uniform_sampling_1D_around_point(1, self.finalAreaCenter,size, self.finalAreaAxis)
        # Use a pertubation approach to get the sample
        deltaX = finishX[0]-self.finalAreaCenter[0]
        deltaY = finishY[0]-self.finalAreaCenter[1]
        return self.sample_path_with_perturbation(deltaX,deltaY)
