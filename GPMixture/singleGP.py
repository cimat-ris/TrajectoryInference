"""
A class for handling a single GP in trajectory prediction
"""
import numpy as np
import math
from regression import *
from evaluation import *
from statistics import*
from sampling import *

class singleGP:

    # Constructor
    def __init__(self, startG, endG, stepUnit, goalsData):
        self.goalsData       = goalsData
        # Goals
        self.nSubgoals       = 2
        self.nPoints         = 5
        n                    = (self.nSubgoals+1)
        self.stepUnit        = stepUnit
        self.startG          = startG
        self.endG            = endG
        self.predictedMeans  = np.empty(n, dtype=object)
        self.predictedVars   = np.empty(n, dtype=object)
        self.sqRootVarX      = np.empty(n, dtype=object)
        self.sqRootVarY      = np.empty(n, dtype=object)
        self.observedX       = None
        self.observedY       = None
        self.observedL       = None
        self.likelihood      = 0.0

    # Update observations and compute likelihood based on observations
    def update(self,observedX,observedY,observedL):
        self.observedX       = observedX
        self.observedY       = observedY
        self.observedL       = observedL
        # Compute likelihood
        self.likelihood = compute_goal_likelihood(self.observedX,self.observedY,self.observedL,self.startG,self.endG,self.nPoints,self.goalsData)
        return self.likelihood

    # Performs prediction
    def predict(self):
        goalCenter = middle_of_area(self.goalsData.areas[self.endG])
        distToGoal = euclidean_distance([self.observedX[-1],self.observedY[-1]], goalCenter)
        dist   = euclidean_distance([self.observedX[0],self.observedY[0]], goalCenter)
        knownN = len(self.observedX)
        # When close to the goal, define sub-goals
        if(distToGoal < 0.4*dist):
            n                    = (self.nSubgoals+1)
            subgoalsCenter, size = get_subgoals_center_and_size(self.nSubgoals, self.goalsData.areas[self.endG], self.goalsData.areasAxis[self.endG])
            self.predictedMeans[0]=None
            self.predictedVars[0] =None
            for j in range(self.nSubgoals):
                print('[INF] Predicting to subgoal ',j)
                predictedX, predictedY, predictedL, varX, varY = prediction_to_finish_point(self.observedX,self.observedY,self.observedL,knownN,subgoalsCenter[j],self.stepUnit,self.startG,self.endG,self.goalsData)
                self.predictedMeans[(j+1)]=[predictedX, predictedY,predictedL]
                # Regularization to avoid singular matrices
                varX = varX + 0.1*np.eye(predictedX.shape[0])
                varY = varY + 0.1*np.eye(predictedY.shape[0])
                self.predictedVars[(j+1)] =[varX, varY]
                # Cholesky on varX
                self.sqRootVarX[(j+1)] = cholesky(varX,lower=True)
                self.sqRootVarY[(j+1)] = cholesky(varY,lower=True)
        # Otherwise, perform prediction to the goal center
        else:
            predictedX, predictedY, predictedL, varX, varY = prediction_to_finish_point(self.observedX,self.observedY,self.observedL,knownN,goalCenter,self.stepUnit,self.startG,self.endG,self.goalsData)
            self.predictedMeans[0]=[predictedX, predictedY,predictedL]
            # Regularization to avoid singular matrices
            varX = varX + 0.1*np.eye(predictedX.shape[0])
            varY = varY + 0.1*np.eye(predictedY.shape[0])
            self.predictedVars[0] =[varX, varY]
            # Cholesky on varX
            self.sqRootVarX[0] = cholesky(varX,lower=True)
            self.sqRootVarY[0] = cholesky(varY,lower=True)
        return self.predictedMeans,self.predictedVars

    def sample(self):
        k          = 0
        # Sample end point around the sampled goal
        if self.predictedMeans[0]!=None:
            finishX, finishY, axis = uniform_sampling_1D(1, self.goalsData.areas[self.endG], self.goalsData.areasAxis[self.endG])
        else:
            # Use subgoals: choose one randomly and sample
            subgoalsCenter, size = get_subgoals_center_and_size(self.nSubgoals, self.goalsData.areas[self.endG], self.goalsData.areasAxis[self.endG])
            if self.goalsData.areasAxis[self.endG]=='x':
                s = size[0]
            else:
                s = size[1]
            # Choose a subgoal randomly: TODO
            k = 1 + np.random.choice(self.nSubgoals)
            #
            finishX, finishY, axis = uniform_sampling_1D_around_point(1, subgoalsCenter[k-1], s, self.goalsData.areasAxis[self.endG])


        # TODO: OPTIMIZE
        # One point at the final of the path
        self.observedX.append(finishX[0])
        self.observedY.append(finishY[0])
        self.observedL.append(self.predictedMeans[k][2][-1])

        # Performs regression for newL
        predictedX, predictedY, varX, varY = prediction_xy(self.observedX,self.observedY,self.observedL,self.predictedMeans[k][2],self.goalsData.kernelsX[self.startG][self.endG],self.goalsData.kernelsY[self.startG][self.endG],self.goalsData.linearPriorsX[self.startG][self.endG],self.goalsData.linearPriorsX[self.startG][self.endG])

        # Removes the last observed point (which was artificially added)
        self.observedX.pop()
        self.observedY.pop()
        self.observedL.pop()

        # Number of predicted points
        nPredictions = len(predictedX)

        # Noise from a normal distribution
        sX = np.random.normal(size=(nPredictions,1))
        sY = np.random.normal(size=(nPredictions,1))
        return predictedX+self.sqRootVarX[k].dot(sX), predictedY+self.sqRootVarY[k].dot(sY)

    def generate_samples(self,nSamples):
        vecX, vecY = [], []
        for k in range(nSamples):
            x, y = self.sample()
            vecX.append(x)
            vecY.append(y)
        return vecX,vecY
