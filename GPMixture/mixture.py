"""
Handling mixtures of GPs
"""
import numpy as np
import math
from regression import *
from evaluation import *
from statistics import*
from sampling import *

class mixtureOfGPs:

    # Constructor
    def __init__(self, startG, stepUnit, goalsData):
        self.goalsData = goalsData
        # Goals
        self.likelyGoals     = []
        self.goalsLikelihood = []
        self.nPoints   = 5
        self.stepUnit  = stepUnit
        self.startG    = startG
        self.nSubgoals = 2
        self.goalsLikelihood = 0.0
        self.observedX = []
        self.observedY = []
        self.observedL = []

    # Update observations and compute likelihoods based on observations
    def update(self,observedX,observedY,observedL):
        self.likelyGoals     = []
        self.goalsLikelihood = []
        self.observedX = observedX
        self.observedY = observedY
        self.observedL = observedL

        for i in range(self.goalsData.nGoals):
            val = compute_goal_likelihood(self.observedX,self.observedY,self.observedL,self.startG,i,self.nPoints,self.goalsData)
            self.goalsLikelihood.append(val)
        # Compute the mean likelihood
        self.meanLikelihood = 0.85*mean(self.goalsLikelihood)
        for i in range(self.goalsData.nGoals):
            if(self.goalsLikelihood[i] > self.meanLikelihood):
                self.likelyGoals.append(i)

    # Performs prediction
    def predict(self):
        plotLikelihood = []
        predictedXYVec, varXYVec = [], []
        # For all likely goals
        for i in range(len(self.likelyGoals)):
            nextG      = self.likelyGoals[i]
            goalCenter = middle_of_area(self.goalsData.areas[nextG])
            distToGoal = euclidean_distance([self.observedX[-1],self.observedY[-1]], goalCenter)
            dist   = euclidean_distance([self.observedX[0],self.observedY[0]], goalCenter)
            knownN = len(self.observedX)
            # When close to the goal, define sub-goals
            if(distToGoal < 0.4*dist):
                subgoalsCenter, size = get_subgoals_center_and_size(self.nSubgoals, self.goalsData.areas[nextG], self.goalsData.areasAxis[nextG])
                for j in range(self.nSubgoals):
                    predictedX, predictedY, predictedL, varX, varY = prediction_to_finish_point(self.observedX,self.observedY,self.observedL,knownN,subgoalsCenter[j],self.stepUnit,self.startG,nextG,self.goalsData)
                    predictedXYVec.append([predictedX, predictedY])
                    varXYVec.append([varX, varY])
                    plotLikelihood.append(self.goalsLikelihood[nextG])
            # Otherwise, perform prediction
            else:
                predictedX, predictedY, predictedL, varX, varY = prediction_to_finish_point(self.observedX,self.observedY,self.observedL,knownN,goalCenter,self.stepUnit,self.startG,nextG,self.goalsData)
                predictedXYVec.append([predictedX, predictedY])
                varXYVec.append([varX, varY])
                plotLikelihood.append(self.goalsLikelihood[nextG])

        return plotLikelihood,predictedXYVec,varXYVec

    def sample(self):
        # Sample goal
        end = 2
        # Sample end point around the sampled goal
        finishX, finishY, axis = uniform_sampling_1D(1, self.goalsData.areas[end], self.goalsData.areasAxis[end])
        # Prediction of the whole trajectory given the
        # start and finish points
        newX, newY, newL, varX, varY = prediction_to_finish_point(self.observedX,self.observedY,self.observedL,len(self.observedL),[finishX[0], finishY[0]],self.stepUnit,self.startG,end,self.goalsData)

        # Number of predicted points
        nPredictions = newX.shape[0]

        # Regularization to avoid singular matrices
        varX = varX + 0.1*np.eye(newX.shape[0])
        varY = varY + 0.1*np.eye(newX.shape[0])
        # Cholesky on varX
        LX = cholesky(varX,lower=True)
        LY = cholesky(varY,lower=True)
        # Noise from a normal distribution
        sX = np.random.normal(size=(nPredictions,1))
        sY = np.random.normal(size=(nPredictions,1))
        return newX+LX.dot(sX), newY+LY.dot(sY), newL, newX, newY

    def generate_samples(self,nSamples):
        vecX, vecY = [], []
        for k in range(nSamples):
            x, y, l, mx, my = self.sample()
            vecX.append(x)
            vecY.append(y)
        return vecX,vecY
