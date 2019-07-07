"""
Handling mixtures of GPs
"""
import numpy as np
import math
from regression import *
from evaluation import *
from statistics import*

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

    # Update likelihoods based on observations
    def update(self,observedX,observedY,observedL):
        self.likelyGoals     = []
        self.goalsLikelihood = []
        for i in range(self.goalsData.nGoals):
            val = compute_goal_likelihood(observedX,observedY,observedL,self.startG,i,self.nPoints,self.goalsData)
            self.goalsLikelihood.append(val)
        # Compute the mean likelihood
        self.meanLikelihood = 0.85*mean(self.goalsLikelihood)
        for i in range(self.goalsData.nGoals):
            if(self.goalsLikelihood[i] > self.meanLikelihood):
                self.likelyGoals.append(i)

        goalCount = 0
        plotLikelihood = []
        predictedXYVec, varXYVec = [], []

        # For all likely goals
        for i in range(len(self.likelyGoals)):
            nextG      = self.likelyGoals[i]
            goalCenter = middle_of_area(self.goalsData.areas[nextG])
            distToGoal = euclidean_distance([observedX[-1],observedY[-1]], goalCenter)
            dist   = euclidean_distance([observedX[0],observedY[0]], goalCenter)
            knownN = len(observedX)
            # When close to the goal, define sub-goals
            if(distToGoal < 0.4*dist):
                subgoalsCenter, size = get_subgoals_center_and_size(self.nSubgoals, self.goalsData.areas[nextG], self.goalsData.areasAxis[nextG])
                for j in range(self.nSubgoals):
                    predictedX, predictedY, predictedL, varX, varY = prediction_to_finish_point(observedX,observedY,observedL,knownN,subgoalsCenter[j],self.stepUnit,self.startG,nextG,self.goalsData)
                    predictedXYVec.append([predictedX, predictedY])
                    varXYVec.append([varX, varY])
                    plotLikelihood.append(self.goalsLikelihood[nextG])
                goalCount += self.nSubgoals
            # Otherwise, perform prediction
            else:
                predictedX, predictedY, predictedL, varX, varY = prediction_to_finish_point(observedX,observedY,observedL,knownN,goalCenter,self.stepUnit,self.startG,nextG,self.goalsData)
                predictedXYVec.append([predictedX, predictedY])
                varXYVec.append([varX, varY])
                plotLikelihood.append(self.goalsLikelihood[nextG])
                goalCount += 1

        return plotLikelihood,predictedXYVec,varXYVec
