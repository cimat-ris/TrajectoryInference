"""
Handling mixtures of GPs in trajectory prediction
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
        self.nSubgoals       = 2
        n                    = (self.nSubgoals+1)*self.goalsData.nGoals
        self.goalsLikelihood = np.empty(n, dtype=object)
        self.nPoints         = 5
        self.stepUnit        = stepUnit
        self.startG          = startG
        self.goalsLikelihood = np.zeros(n, dtype=float)
        self.predictedMeans  = np.empty(n, dtype=object)
        self.predictedVars   = np.empty(n, dtype=object)
        self.sqRootVarX      = np.empty(n, dtype=object)
        self.sqRootVarY      = np.empty(n, dtype=object)
        self.observedX       = None
        self.observedY       = None
        self.observedL       = None

    # Update observations and compute likelihoods based on observations
    def update(self,observedX,observedY,observedL):
        self.observedX       = observedX
        self.observedY       = observedY
        self.observedL       = observedL
        # Compute likelihoods
        for i in range(self.goalsData.nGoals):
            val = compute_goal_likelihood(self.observedX,self.observedY,self.observedL,self.startG,i,self.nPoints,self.goalsData)
            self.goalsLikelihood[i] = val
        # Sum of the likelihoods
        s = sum(self.goalsLikelihood)
        # Compute the mean likelihood
        self.meanLikelihood = mean(self.goalsLikelihood)
        # Maintain a list of likely goals
        for i in range(self.goalsData.nGoals):
            self.goalsLikelihood[i] /= s
            if(self.goalsLikelihood[i] > 0.85*self.meanLikelihood):
                self.likelyGoals.append(i)
        return self.goalsLikelihood

    # Performs prediction
    def predict(self):
        # For all likely goals
        for nextG in range(self.goalsData.nGoals):
            print('[INF] Predicting to goal ',nextG)
            goalCenter = middle_of_area(self.goalsData.areas[nextG])
            distToGoal = euclidean_distance([self.observedX[-1],self.observedY[-1]], goalCenter)
            dist   = euclidean_distance([self.observedX[0],self.observedY[0]], goalCenter)
            knownN = len(self.observedX)
            # When close to the goal, define sub-goals
            if(distToGoal < 0.4*dist):
                n                    = (self.nSubgoals+1)*self.goalsData.nGoals
                subgoalsCenter, size = get_subgoals_center_and_size(self.nSubgoals, self.goalsData.areas[nextG], self.goalsData.areasAxis[nextG])
                self.predictedMeans[nextG]=None
                self.predictedVars[nextG] =None
                for j in range(self.nSubgoals):
                    print('[INF] Predicting to subgoal ',j)
                    predictedX, predictedY, predictedL, varX, varY = prediction_to_finish_point(self.observedX,self.observedY,self.observedL,knownN,subgoalsCenter[j],self.stepUnit,self.startG,nextG,self.goalsData)
                    self.predictedMeans[(j+1)*self.goalsData.nGoals+nextG]=[predictedX, predictedY,predictedL]
                    # Regularization to avoid singular matrices
                    varX = varX + 0.1*np.eye(predictedX.shape[0])
                    varY = varY + 0.1*np.eye(predictedY.shape[0])
                    self.predictedVars[(j+1)*self.goalsData.nGoals+nextG] =[varX, varY]
                    # Cholesky on varX
                    self.sqRootVarX[(j+1)*self.goalsData.nGoals+nextG] = cholesky(varX,lower=True)
                    self.sqRootVarY[(j+1)*self.goalsData.nGoals+nextG] = cholesky(varY,lower=True)
            # Otherwise, perform prediction
            else:
                predictedX, predictedY, predictedL, varX, varY = prediction_to_finish_point(self.observedX,self.observedY,self.observedL,knownN,goalCenter,self.stepUnit,self.startG,nextG,self.goalsData)
                self.predictedMeans[nextG]=[predictedX, predictedY,predictedL]
                # Regularization to avoid singular matrices
                varX = varX + 0.1*np.eye(predictedX.shape[0])
                varY = varY + 0.1*np.eye(predictedY.shape[0])
                self.predictedVars[nextG] =[varX, varY]
                # Cholesky on varX
                self.sqRootVarX[nextG] = cholesky(varX,lower=True)
                self.sqRootVarY[nextG] = cholesky(varY,lower=True)
        return self.predictedMeans,self.predictedVars

    def sample(self):
        p = self.goalsLikelihood[:self.goalsData.nGoals]
        # Sample goal
        goalSample = np.random.choice(self.goalsData.nGoals,1,p=p)
        end        = goalSample[0]
        k          = 0
        # Sample end point around the sampled goal
        if self.predictedMeans[end]!=None:
            finishX, finishY, axis = uniform_sampling_1D(1, self.goalsData.areas[end], self.goalsData.areasAxis[end])
        else:
            # Use subgoals: choose one randomly and sample
            subgoalsCenter, size = get_subgoals_center_and_size(self.nSubgoals, self.goalsData.areas[end], self.goalsData.areasAxis[end])
            if self.goalsData.areasAxis[end]=='x':
                s = size[0]
            else:
                s = size[1]
            # Choose a subgoal randomly
            k = 1 + np.random.choice(self.nSubgoals)
            #
            finishX, finishY, axis = uniform_sampling_1D_around_point(1, subgoalsCenter[k-1], s, self.goalsData.areasAxis[end])


        # TODO: OPTIMIZE
        # One point at the final of the path
        self.observedX.append(finishX[0])
        self.observedY.append(finishY[0])
        index = end+k*self.goalsData.nGoals
        self.observedL.append(self.predictedMeans[index][2][-1])

        # Performs regression for newL
        predictedX, predictedY, varX, varY = prediction_xy(self.observedX,self.observedY,self.observedL,self.predictedMeans[index][2],self.goalsData.kernelsX[self.startG][end],self.goalsData.kernelsY[self.startG][end],self.goalsData.linearPriorsX[self.startG][end],self.goalsData.linearPriorsX[self.startG][end])

        # Removes the last observed point (which was artificially added)
        self.observedX.pop()
        self.observedY.pop()
        self.observedL.pop()

        # Number of predicted points
        nPredictions = len(predictedX)

        # Noise from a normal distribution
        sX = np.random.normal(size=(nPredictions,1))
        sY = np.random.normal(size=(nPredictions,1))
        return predictedX+self.sqRootVarX[index].dot(sX), predictedY+self.sqRootVarY[index].dot(sY)

    def generate_samples(self,nSamples):
        vecX, vecY = [], []
        for k in range(nSamples):
            x, y = self.sample()
            vecX.append(x)
            vecY.append(y)
        return vecX,vecY
