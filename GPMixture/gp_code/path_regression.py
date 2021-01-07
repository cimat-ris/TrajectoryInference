"""
A class for GP-based path regression
"""
import numpy as np
import math
from gp_code.regression import *
from gp_code.path1D_regression import path1D_regression
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

        maxDim = 20

        n = len(observedX)
        if n > maxDim:
            print("[INF] Sampling data")
            ind = [i for i in range(1,n)]
            p = 0.5
            iprob = [ pow((1-p),n-i)*p for i in ind ]
            iprob[0] = 1- (sum(iprob) -iprob[0])

            #sampleInd = np.random.choice(ind, maxDim-1, replace=False, p=iprob) #geometric
            sampleInd = np.random.choice(ind, maxDim-1, replace=False)         #normal

            observedL = np.append(observedL[0], [observedL[i] for i in np.sort(sampleInd)])
            observedX = np.append(observedX[0], [observedX[i] for i in np.sort(sampleInd)])
            observedY = np.append(observedY[0], [observedY[i] for i in np.sort(sampleInd)])

        # Update observations of each process
        self.regression_x.updateObservations(observedX,observedL,self.finalAreaCenter[0],finalL,(1.0-self.finalAreaAxis)*s*s*math.exp(-self.dist/s),self.predictedL)
        self.regression_y.updateObservations(observedY,observedL,self.finalAreaCenter[1],finalL,    (self.finalAreaAxis)*s*s*math.exp(-self.dist/s),self.predictedL)

    # Filter initial observations
    def filterObservations(self):
        filteredx = self.regression_x.filterObservations()
        filteredy = self.regression_y.filterObservations()
        return filteredx,filteredy

    # For a given dataset (knownX,knownY,knownL), takes half of the data as known
    # and predicts the remaining half. Then, evaluate the prediction error.
    def compute_prediction_error_of_points_along_the_path(self,nPoints,observedX,observedY,observedL,startG,finishG,goalsData):
        # Known data
        observedN = len(observedX)
        halfN     = max(1,int(observedN/2))
        # First half of the known data
        trueX  = observedX[0:halfN]
        trueY  = observedY[0:halfN]
        trueL  = observedL[0:halfN]
        # Get the last point and add it to the observed data
        finishD= euclidean_distance([trueX[len(trueX)-1],trueY[len(trueY)-1]],self.finalAreaCenter)
        np.append(trueX,self.finalAreaCenter[0])
        np.append(trueY,self.finalAreaCenter[1])
        np.append(trueL,finishD*goalsData.units[startG][finishG])
        d = int(halfN/nPoints)
        if d<1:
            return 1.0
        return 1.0
        # Prepare the ground truths and the list of l to evaluate
        realX         = observedX[halfN:halfN+nPoints*d:d]
        realY         = observedY[halfN:halfN+nPoints*d:d]
        predictionSet = observedL[halfN:halfN+nPoints*d:d]
        # TODO! redo the prediction based on GP model
        # Get the prediction based on the GP
        # Evaluate the error
        # return ADE([realX,realY],[predX,predY])


    # For a given set of observations (observedX,observedY,observedL),
    # takes half of the data to fit a model and predicts the remaining half. Then, evaluate the likelihood.
    def likelihood_from_partial_path(self,nPoints,observedX,observedY,observedL,startG,finishG,goalsData):
        D = 150. #value for compute_goal_likelihood
        error = self.compute_prediction_error_of_points_along_the_path(nPoints,observedX,observedY,observedL,startG,finishG,goalsData)
        return (math.exp(-1.*( error**2)/D**2 ))

    # Compute the likelihood
    def computeLikelihood(self,observedX,observedY,observedL,startG,finishG,stepsToCompare,goalsData):
        # TODO: remove the goalsData structure
        self.likelihood = goalsData.priorTransitions[startG][finishG]*self.likelihood_from_partial_path(stepsToCompare,observedX,observedY,observedL,startG,finishG,goalsData)
        return self.likelihood

    # The main path regression function: perform regression for a
    # vector of values of future L, that has been computed in update
    def prediction_to_finish_point(self,compute_sqRoot=False):
        pL,pX,vX = self.regression_x.prediction_to_finish_point(compute_sqRoot=compute_sqRoot)
        pL,pY,vY = self.regression_y.prediction_to_finish_point(compute_sqRoot=compute_sqRoot)
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
