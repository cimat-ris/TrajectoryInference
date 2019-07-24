"""
Handling mixtures of GPs in trajectory prediction
"""
import numpy as np
import math
from gp_code.regression import *
from gp_code.evaluation import *
from gp_code.sampling import *
from gp_code.gpRegressor import *
from statistics import mean

# Class for performing path regression with a mixture of Gaussian processes
class mixtureOfGPs:

    # Constructor
    def __init__(self, startG, stepUnit, goalsData):
        self.goalsData       = goalsData
        # Goals
        self.likelyGoals     = []
        self.nSubgoals       = 2
        n                    = (self.nSubgoals+1)*self.goalsData.nGoals
        self.goalsLikelihood = np.empty(n, dtype=object)
        self.nPoints         = 5
        self.stepUnit        = stepUnit
        self.startG          = startG
        self.goalsLikelihood = np.zeros(n, dtype=float)
        self.predictedMeans  = [np.zeros((0,3), dtype=float)]*n
        self.predictedVars   = [np.zeros((0,0,0), dtype=float)]*n
        self.sqRootVarX      = np.empty(n, dtype=object)
        self.sqRootVarY      = np.empty(n, dtype=object)
        self.observedX       = None
        self.observedY       = None
        self.observedL       = None
        # The basic element here is this object, that will do the regression work
        self.gpPathRegressor = [None]*n
        for i in range(self.goalsData.nGoals):
            # One regressor per goal
            self.gpPathRegressor[i] = gpRegressor(self.goalsData.kernelsX[self.startG][i], self.goalsData.kernelsY[self.startG][i],goalsData.units[self.startG][i],stepUnit,self.goalsData.areas[i],self.goalsData.areasAxis[i],self.goalsData.linearPriorsX[self.startG][i], self.goalsData.linearPriorsY[self.startG][i])

            ## TODO:
            subareas = get_subgoals_areas(self.nSubgoals, self.goalsData.areas[i],self.goalsData.areasAxis[i])
            # For sub-goals
            for j in range(self.nSubgoals):
                k= i+(j+1)*self.goalsData.nGoals
                self.gpPathRegressor[k] = gpRegressor(self.goalsData.kernelsX[self.startG][i],self.goalsData.kernelsY[self.startG][i],goalsData.units[self.startG][i],stepUnit,subareas[j],self.goalsData.areasAxis[i],self.goalsData.linearPriorsX[self.startG][i],self.goalsData.linearPriorsY[self.startG][i])



    # Update observations and compute likelihoods based on observations
    def update(self,observedX,observedY,observedL):
        self.observedX       = observedX
        self.observedY       = observedY
        self.observedL       = observedL

        # Update each regressor with its corresponding observations
        for i in range(self.goalsData.nGoals):
            goalCenter = middle_of_area(self.goalsData.areas[i])
            distToGoal = euclidean_distance([self.observedX[-1],self.observedY[-1]], goalCenter)
            dist       = euclidean_distance([self.observedX[0],self.observedY[0]], goalCenter)
            # When close to the goal, define sub-goals
            if(distToGoal < 0.4*dist):
                for j in range(self.nSubgoals):
                    print('[INF] Updating subgoal ',j)
                    k= i+(j+1)*self.goalsData.nGoals
                    self.gpPathRegressor[k].updateObservations(observedX,observedY,observedL)
            else:
                # Update observations and re-compute the kernel matrices
                self.gpPathRegressor[i].updateObservations(observedX,observedY,observedL)
            # Compute the model likelihood
            self.goalsLikelihood[i] = self.gpPathRegressor[i].computeLikelihood(observedX,observedY,observedL,self.startG,i,self.nPoints,self.goalsData)

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
        for i in range(self.goalsData.nGoals):
            print('[INF] Predicting to goal ',i)
            goalCenter = middle_of_area(self.goalsData.areas[i])
            distToGoal = euclidean_distance([self.observedX[-1],self.observedY[-1]], goalCenter)
            dist       = euclidean_distance([self.observedX[0],self.observedY[0]], goalCenter)
            knownN = len(self.observedX)

            # When close to the goal, define sub-goals
            if(distToGoal < 0.4*dist):
                self.predictedMeans[i]=np.zeros((0,3), dtype=float)
                self.predictedVars[i]=np.zeros((0,0,0), dtype=float)
                for j in range(self.nSubgoals):
                    print('[INF] Predicting to subgoal ',j)
                    k= i+(j+1)*self.goalsData.nGoals
                    predictedX, predictedY, predictedL, varX, varY = self.gpPathRegressor[k].prediction_to_finish_point()
                    self.predictedMeans[k]=np.column_stack((predictedX, predictedY, predictedL))
                    self.predictedVars[k] = np.stack([varX, varY],axis=0)
            # Otherwise, perform prediction
            else:
                # Uses the already computed matrices to apply regression over missing data
                predictedX, predictedY, predictedL, varX, varY = self.gpPathRegressor[i].prediction_to_finish_point()
                self.predictedMeans[i] = np.column_stack((predictedX, predictedY, predictedL))
                self.predictedVars[i]  = np.stack([varX, varY],axis=0)

        return self.predictedMeans,self.predictedVars


    def sample(self):
        p = self.goalsLikelihood[:self.goalsData.nGoals]
        # Sample goal
        goalSample = np.random.choice(self.goalsData.nGoals,1,p=p)
        end        = goalSample[0]
        k          = end
        # Sample end point around the sampled goal
        if self.predictedMeans[end].shape[0]>0:
            finishX, finishY, axis = uniform_sampling_1D(1, self.goalsData.areas[end], self.goalsData.areasAxis[end])
        else:
            self.update(self.observedX,self.observedY,self.observedL)
            # Use subgoals: choose one randomly and sample
            subgoalsCenter, size = get_subgoals_center_and_size(self.nSubgoals, self.goalsData.areas[end], self.goalsData.areasAxis[end])
            if self.goalsData.areasAxis[end]==0:
                s = size[0]
            else:
                s = size[1]
            # Choose a subgoal randomly
            j = np.random.choice(self.nSubgoals)
            k = end+(1+j)*self.goalsData.nGoals
            self.gpPathRegressor[k].updateObservations(self.observedX,self.observedY,self.observedL) #we call this in case the subgoals haven't been updated
            #
            finishX, finishY, axis = uniform_sampling_1D_around_point(1, subgoalsCenter[j], s, self.goalsData.areasAxis[end])


        # Use a pertubation approach to get the sample
        goalCenter = middle_of_area(self.goalsData.areas[end])
        deltaX = finishX[0]-goalCenter[0]
        deltaY = finishY[0]-goalCenter[1]

        return self.gpPathRegressor[k].sample_with_perturbation(deltaX,deltaY)

    def generate_samples(self,nSamples):
        vecX, vecY = [], []
        for k in range(nSamples):
            x, y = self.sample()
            vecX.append(x)
            vecY.append(y)
        return vecX,vecY
