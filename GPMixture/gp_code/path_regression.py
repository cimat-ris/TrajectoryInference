"""
A class for GP-based path regression
"""
import numpy as np
import math
from gp_code.regression import *
from gp_code.sampling import *
from gp_code.path1D_regression import path1D_regression
from utils.manip_trajectories import goal_center_and_size
from utils.stats_trajectories import euclidean_distance
from gp_code.likelihood import ADE
from scipy.optimize import bisect

class path_regression:
    # Constructor
    def __init__(self, kernelX, kernelY, unit, stepUnit, finalArea, finalAreaAxis, prior):
        self.regression_x    = path1D_regression(kernelX)
        self.regression_y    = path1D_regression(kernelY)
        self.predictedL      = None
        self.distUnit        = unit
        self.stepUnit        = stepUnit
        self.finalArea       = finalArea
        self.finalAreaAxis   = finalAreaAxis
        self.finalAreaCenter, self.finalAreaSize = goal_center_and_size(finalArea)
        self.prior           = prior

    # Update observations for the Gaussian process (matrix K)
    def updateObservations(self,observedX,observedY,observedL):
        # Last really observed point
        lastObs = [observedX[-1], observedY[-1], observedL[-1]]
        # Determine the set of arclengths (predictedL) to predict
        self.predictedL, finalL, self.dist = self.prediction_set_arclength(lastObs,self.finalAreaCenter)
        # Define the variance associated to the last point (varies with the area)
        if self.finalAreaAxis==0:
            s              = self.finalAreaSize[0]
        elif self.finalAreaAxis==1:
            s              = self.finalAreaSize[1]

        # Update observations of each process
        self.regression_x.updateObservations(observedX,observedL,self.finalAreaCenter[0],finalL,(1.0-self.finalAreaAxis)*s*s*math.exp(-self.dist/s),self.predictedL)
        self.regression_y.updateObservations(observedY,observedL,self.finalAreaCenter[1],finalL,    (self.finalAreaAxis)*s*s*math.exp(-self.dist/s),self.predictedL)

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
    def prediction_set_time(lastObs, elapsedTime, timeTransitionData, timeStep):
        # Time of the last observed point
        t = lastObs[2]
        # TODO: I think we should first do here a fully deterministic model (conditioned on the mean transition time)
        # Sample a duration
        transitionTime = int(np.random.normal(timeTransitionData[0], timeTransitionData[1]) )
        # Remaining time
        remainingTime = transitionTime - elapsedTime
        size = int(remainingTime/timeStep)
        predset = np.zeros((size,1))
        if size > 0:
            for i in range(1,size+1):
                predset[i-1,0] = t + i*timeStep
            if predset[-1,0] < t + remainingTime:
                predset[-1,0] = t + remainingTime
        
        return predset, t + remainingTime, remainingTime
    
    # Filter initial observations
    def filterObservations(self):
        filteredx = self.regression_x.filterObservations()
        filteredy = self.regression_y.filterObservations()
        return filteredx,filteredy

    # For a given set of observations (x,y,l), takes half of the data as known
    # and predicts m points from the remaining half. Then, evaluate the prediction error.
    def likelihood_from_partial_path(self,m,observedX,observedY,observedL):
        n       = len(observedX)
        half    = max(1,int(n/2))
        lastObs = [observedX[half], observedY[half], observedL[half]]
        _, finalL, self.dist = self.prediction_set_arclength(lastObs,self.finalAreaCenter)
        # Set of l to predict
        d = int(half/m)
        if d<1:
            return 1.0
        predset = observedL[half:half+m*d:d]
        self.predictedL = np.zeros((m,1))
        for i in range(m):
            self.predictedL[i,0] = predset[i]
        # Define the variance associated to the last point (varies with the area)
        if self.finalAreaAxis==0:
            s              = self.finalAreaSize[0]
        elif self.finalAreaAxis==1:
            s              = self.finalAreaSize[1]
        # Update observations of each process
        self.regression_x.updateObservations(observedX[:half],observedL[:half],self.finalAreaCenter[0],finalL,(1.0-self.finalAreaAxis)*s*s*math.exp(-self.dist/s),self.predictedL)
        self.regression_y.updateObservations(observedY[:half],observedL[:half],self.finalAreaCenter[1],finalL,    (self.finalAreaAxis)*s*s*math.exp(-self.dist/s),self.predictedL)

        predX, predY, _, _, _ = self.prediction_to_finish_point()
        # Prepare the ground truths
        trueX           = observedX[half:half+m*d:d]
        trueY           = observedY[half:half+m*d:d]
        # Return to original values
        self.updateObservations(observedX, observedY, observedL)
        # Compute Average Displacement Error between prediction and true values
        D = 150.
        error = ADE([trueX,trueY],[predX.reshape(m),predY.reshape(m)])
        return (math.exp(-1.*( error**2)/D**2 ))

    # Compute the likelihood
    def computeLikelihood(self,observedX,observedY,observedL,stepsToCompare):
        self.likelihood = self.prior*self.likelihood_from_partial_path(stepsToCompare,observedX,observedY,observedL)
        # Another option 
        #self.likelihood = self.prior*self.regression_x.compute_likelihood()*self.regression_y.compute_likelihood()
        return self.likelihood

    # The main path regression function: perform regression for a
    # vector of values of future L, that has been computed in update
    def prediction_to_finish_point(self,compute_sqRoot=False):
        predx, varx = self.regression_x.prediction_to_finish_point(compute_sqRoot=compute_sqRoot)
        predy, vary = self.regression_y.prediction_to_finish_point(compute_sqRoot=compute_sqRoot)
        return predx, predy, self.predictedL, varx, vary

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
