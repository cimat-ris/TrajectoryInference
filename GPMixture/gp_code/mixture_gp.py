"""
Handling mixtures of GPs in trajectory prediction
"""
import numpy as np
import math
from gp_code.regression import *
from gp_code.sampling import *
from gp_code.trajectory_regression import *
from gp_code.likelihood import nearestPD
from utils.stats_trajectories import euclidean_distance
from utils.manip_trajectories import goal_center_and_size
from statistics import mean

# Class for performing path regression with a mixture of Gaussian processes
class mixtureOfGPs:

    # Constructor
    def __init__(self, startG, goalsData):
        # The goals structure
        self.goalsData       = goalsData
        # Sub-set of likely goals
        self.likelyGoals     = []
        #Index of most likely goal
        self.mostLikelyGoal = None
        # Number of sub-goals
        self.nSubgoals       = 2
        # Number of elements in the mixture (not all are used at the same time)
        n                    = (self.nSubgoals+1)*self.goalsData.nGoals
        # Points to evaluate the likelihoods
        self.nPoints         = 5
        # Step unit
        self.stepUnit        = goalsData.stepUnit
        # Starting goal
        self.startG          = startG
        # Likelihoods
        self.goalsLikelihood = np.zeros(n, dtype=float)
        # Predicted means (per element of the mixture)
        self.predictedMeans  = [np.zeros((0,3), dtype=float)]*n
        self.predictedVars   = [np.zeros((0,0,0), dtype=float)]*n
        self.sqRootVarX      = np.empty(n, dtype=object)
        self.sqRootVarY      = np.empty(n, dtype=object)
        self.observedX       = None
        self.observedY       = None
        self.observedL       = None
        # The basic element here is this object, that will do the regression work
        self.gpPathRegressor = [None]*n
        self.gpTrajectoryRegressor = [None]*n
        for i in range(self.goalsData.nGoals):
            # One regressor per goal
            self.gpPathRegressor[i] = trajectory_regression(self.goalsData.kernelsX[self.startG][i], self.goalsData.kernelsY[self.startG][i],self.goalsData.speedModels[self.startG][i],self.goalsData.units[self.startG][i],self.stepUnit,self.goalsData.areas_coordinates[i],self.goalsData.areas_axis[i],self.goalsData.priorTransitions[self.startG][i])

    # Update observations and compute likelihoods based on observations
    def update(self,observations):
        self.observedX = observations[:,0]
        self.observedY = observations[:,1]
        self.observedT = observations[:,2]
        # Update each regressor with its corresponding observations
        for i in range(self.goalsData.nGoals):
            goalCenter,__= goal_center_and_size(self.goalsData.areas_coordinates[i])
            distToGoal   = euclidean_distance([self.observedX[-1],self.observedY[-1]], goalCenter)
            dist         = euclidean_distance([self.observedX[0],self.observedY[0]], goalCenter)
            # Update observations and re-compute the kernel matrices
            self.gpPathRegressor[i].update_observations(observations)
            # Compute the model likelihood
            self.goalsLikelihood[i] = self.gpPathRegressor[i].compute_likelihood(observations,self.nPoints)

        # Sum of the likelihoods
        s = sum(self.goalsLikelihood)
        # Compute the mean likelihood
        self.meanLikelihood = mean(self.goalsLikelihood)
        # Maintain a list of likely goals
        for i in range(self.goalsData.nGoals):
            self.goalsLikelihood[i] /= s
            if(self.goalsLikelihood[i] > 0.85*self.meanLikelihood):
                self.likelyGoals.append(i)
        # Save most likely goal
        mostLikely = 0
        for i in range(self.goalsData.nGoals):
            if self.goalsLikelihood[i] > self.goalsLikelihood[mostLikely]:
                mostLikely = i
        self.mostLikelyGoal = mostLikely

        return self.goalsLikelihood

    # Performs prediction
    def predict_path(self):
        # For all likely goals
        for i in range(self.goalsData.nGoals):
            print('[INF] Predicting to goal ',i)
            goalCenter,__ = goal_center_and_size(self.goalsData.areas_coordinates[i])
            distToGoal    = euclidean_distance([self.observedX[-1],self.observedY[-1]], goalCenter)
            dist          = euclidean_distance([self.observedX[0],self.observedY[0]], goalCenter)

            # When close to the goal, define sub-goals
            # Uses the already computed matrices to apply regression over missing data
            self.predictedMeans[i], self.predictedVars[i] = self.gpPathRegressor[i].predict_path_to_finish_point()

        return self.predictedMeans,self.predictedVars


    def sample_path(self):
        p = self.goalsLikelihood[:self.goalsData.nGoals]
        # Sample goal
        goalSample = np.random.choice(self.goalsData.nGoals,1,p=p)
        end        = goalSample[0]
        k          = end
        # Sample end point around the sampled goal
        if self.predictedMeans[end].shape[0]>0:
            finishX, finishY, axis = uniform_sampling_1D(1, self.goalsData.areas_coordinates[end], self.goalsData.areas_axis[end])
        else:
            # Use subgoals: choose one randomly and sample
            subgoalsCenter, size = get_subgoals_center_and_size(self.nSubgoals, self.goalsData.areas_coordinates[end], self.goalsData.areas_axis[end])
            if self.goalsData.areas_axis[end]==0:
                s = size[0]
            else:
                s = size[1]
            # Choose a subgoal randomly
            j = np.random.choice(self.nSubgoals)
            k = end+(1+j)*self.goalsData.nGoals
            # We call this in case the subgoals haven't been updated
            finishX, finishY, axis = uniform_sampling_1D_around_point(1, subgoalsCenter[j], s, self.goalsData.areas_axis[end])

        # Use a pertubation approach to get the sample
        deltaX = finishX[0]-self.gpPathRegressor[k].finalAreaCenter[0]
        deltaY = finishY[0]-self.gpPathRegressor[k].finalAreaCenter[1]
        return self.gpPathRegressor[k].sample_path_with_perturbation(deltaX,deltaY)

    def sample_paths(self,nSamples):
        vecX, vecY, vecL = [], [], []
        for k in range(nSamples):
            x, y, l = self.sample_path()
            vecX.append(x)
            vecY.append(y)
            vecL.append(l)
        return vecX,vecY,vecL
