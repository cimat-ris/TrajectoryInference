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
        # Number of elements in the mixture
        n                    = self.goalsData.goals_n
        # TODO: update?
        # Points to evaluate the likelihoods
        self.nPoints         = 5
        # Step unit
        self.stepUnit        = goalsData.stepUnit
        # Starting goal
        self.startG          = startG
        # Likelihoods
        self.goalsLikelihood = np.zeros(n, dtype=float)
        # Predicted means (per element of the mixture)
        self.predictedMeans  = [np.zeros((0,3),  dtype=float)]*n
        self.predictedVars   = [np.zeros((0,0,0),dtype=float)]*n
        self.filteredPaths   = [np.zeros((0,3),  dtype=float)]*n
        self.sqRootVarX      = np.empty(n, dtype=object)
        self.sqRootVarY      = np.empty(n, dtype=object)
        self.observedX       = None
        self.observedY       = None
        self.observedL       = None
        # The basic elements here is this array of objects, that will do the regression work
        self.gpTrajectoryRegressor = [None]*n
        for i in range(self.goalsData.goals_n):
            # One regressor per goal
            self.gpTrajectoryRegressor[i]=trajectory_regression(self.goalsData.kernelsX[self.startG][i], self.goalsData.kernelsY[self.startG][i],goalsData.sigmaNoise,self.goalsData.speedModels[self.startG][i],self.goalsData.units[self.startG][i],self.goalsData.stepUnit,self.goalsData.goals_areas[i],prior=self.goalsData.priorTransitions[self.startG][i])

    # Update observations and compute likelihoods based on observations
    def update(self,observations):
        self.observedX = observations[:,0]
        self.observedY = observations[:,1]
        self.observedT = observations[:,2]
        # Update each regressor with its corresponding observations
        for i in range(self.goalsData.goals_n):
            goalCenter,__= goal_center_and_size(self.goalsData.goals_areas[i][1:])
            distToGoal   = euclidean_distance([self.observedX[-1],self.observedY[-1]], goalCenter)
            dist         = euclidean_distance([self.observedX[0],self.observedY[0]], goalCenter)
            # Update observations and re-compute the kernel matrices
            self.gpTrajectoryRegressor[i].update_observations(observations)
            # Compute the model likelihood
            self.goalsLikelihood[i] = self.gpTrajectoryRegressor[i].compute_likelihood(observations,self.nPoints)

        # Sum of the likelihoods
        s = sum(self.goalsLikelihood)
        # Compute the mean likelihood
        self.meanLikelihood = mean(self.goalsLikelihood)
        # Maintain a list of likely goals
        for i in range(self.goalsData.goals_n):
            self.goalsLikelihood[i] /= s
            if(self.goalsLikelihood[i] > 0.85*self.meanLikelihood):
                self.likelyGoals.append(i)
        # Save most likely goal
        mostLikely = 0
        for i in range(self.goalsData.goals_n):
            if self.goalsLikelihood[i] > self.goalsLikelihood[mostLikely]:
                mostLikely = i
        self.mostLikelyGoal = mostLikely

        return self.goalsLikelihood

    # Get a filtered version of the initial observations
    def filter(self):
        # For all likely goals
        for i in range(self.goalsData.goals_n):
            self.filteredPaths[i]=self.gpTrajectoryRegressor[i].filter_observations()
        return self.filteredPaths

    # Performs path prediction
    def predict_path(self,compute_sqRoot=False):
        # For all likely goals
        for i in range(self.goalsData.goals_n):
            print('[INF] Predicting path to goal ',i)
            # Uses the already computed matrices to apply regression over missing data
            self.predictedMeans[i], self.predictedVars[i] = self.gpTrajectoryRegressor[i].predict_path_to_finish_point(compute_sqRoot=compute_sqRoot)

        return self.predictedMeans,self.predictedVars

    # Performs trajectory prediction
    def predict_trajectory(self,compute_sqRoot=False):
        # For all likely goals
        for i in range(self.goalsData.goals_n):
            print('[INF] Predicting trajectory to goal ',i)
            # Uses the already computed matrices to apply regression over missing data
            self.predictedMeans[i], self.predictedVars[i] = self.gpTrajectoryRegressor[i].predict_trajectory_to_finish_point(compute_sqRoot=compute_sqRoot)
        return self.predictedMeans,self.predictedVars

    def sample_path(self):
        p = self.goalsLikelihood[:self.goalsData.goals_n]
        # Sample goal
        goalSample = np.random.choice(self.goalsData.goals_n,1,p=p)
        end        = goalSample[0]
        k          = end
        finishX, finishY, axis = uniform_sampling_1D(1, self.goalsData.goals_areas[end][1:], self.goalsData.goals_areas[end][0])
        # Use a pertubation approach to get the sample
        deltaX = finishX[0]-self.gpTrajectoryRegressor[k].finalAreaCenter[0]
        deltaY = finishY[0]-self.gpTrajectoryRegressor[k].finalAreaCenter[1]
        return self.gpTrajectoryRegressor[k].sample_path_with_perturbation(deltaX,deltaY)

    # Generate samples from the predictive distribution
    def sample_paths(self,nSamples):
        vec = []
        for k in range(nSamples):
            path = self.sample_path()
            vec.append(path)
        return vec
