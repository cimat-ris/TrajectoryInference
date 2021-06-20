"""
Handling mixtures of GPs in trajectory prediction | Trautman
"""
import numpy as np
from statistics import mean
from gp_code.sampling import *
from gp_code.path_regression import path_regression
from gp_code.likelihood import nearestPD
from utils.stats_trajectories import euclidean_distance
from utils.manip_trajectories import goal_center_and_size

# Class for performing path regression with a mixture of Gaussian processes with time variable (Trautman's approach)
class mGPt_trajectory_prediction:
    def __init__(self, startG, goalsData):
        # The goals structure
        self.goalsData       = goalsData
        # Sub-set of likely goals
        self.likelyGoals     = []
        #Index of most likely goal
        self.mostLikelyGoal = None
        # Max number of elements in the mixture (not all are used at the same time)
        maxn = 5#10
        # Array of potential future goals
        arr = np.random.choice([i for i in range(goalsData.goals_n)], maxn, replace=False, p=goalsData.priorTransitions[startG])
        # Select elements where timeTransition is not zero
        deleteid = []
        for i in range(maxn):
            if goalsData.timeTransitionMeans[startG][arr[i]] == 0:
                deleteid.append(i)
        self.goalTransitions = np.delete(arr, deleteid)
        n                    = self.goalTransitions.size
        # Points to evaluate the likelihoods
        self.nPoints         = 5
        # Starting goal
        self._start          = startG
        # Likelihoods
        self._goals_likelihood= np.zeros(n, dtype=float)
        # Predicted means (per element of the mixture)
        self._predicted_means = [np.zeros((0,3), dtype=float)]*n
        self._predicted_vars  = [np.zeros((0,0,0), dtype=float)]*n
        self._observed_x      = None
        self._observed_y      = None
        self._observed_l      = None
        # The basic element here is this object, that will do the regression work
        self.gpPathRegressor = [None]*n
        self.gpTrajectoryRegressor = [None]*n

        for i in range(n):
            gi = self. goalTransitions[i]
            timeTransitionData = [self.goalsData.timeTransitionMeans[self._start][gi],self.goalsData.timeTransitionStd[self._start][gi]]
            self.gpPathRegressor[i] = path_regression(self.goalsData.kernelsX[self._start][gi], self.goalsData.kernelsY[self._start][gi],goalsData.sigmaNoise,None,self.goalsData.goals_areas[gi],mode='Trautman',timeTransitionData=timeTransitionData)

    def update(self, observations):
        self._observed_x = observations[:,0]
        self._observed_y = observations[:,1]
        self._observed_t = observations[:,2]
        # Update each regressor with its corresponding observations
        for i in range(len(self.goalTransitions)):
            # Update observations and re-compute the kernel matrices
            self.gpPathRegressor[i].update_observations(observations)
            # Compute the model likelihood
            self._goals_likelihood[i] = self.gpPathRegressor[i].compute_likelihood()

        # Compute the mean likelihood
        self.meanLikelihood = mean(self._goals_likelihood)

        n = len(self.goalTransitions)
        mostLikely = 0
        # TODO: avoid cycle
        for i in range(n):
            if self._goals_likelihood[i] > self._goals_likelihood[mostLikely]:
                mostLikely = i
        self.mostLikelyGoal = mostLikely

        return self._goals_likelihood

    # Performs prediction
    def predict_path(self):
        n = len(self.goalTransitions)
        # For all likely goals
        for i in range(n):
            gi = self.goalTransitions[i]
            goalCenter,__ = goal_center_and_size(self.goalsData.goals_areas[gi,1:])
            # Uses the already computed matrices to apply regression over missing data
            self._predicted_means[i], self._predicted_vars[i] = self.gpPathRegressor[i].predict_path_to_finish_point()

        return self._predicted_means,self._predicted_vars

    def sample_path(self):
        n = len(self.goalTransitions)
        p = self._goals_likelihood[:n]
        normp = p/np.linalg.norm(p,ord=1)
        # Sample goal
        sampleId = np.random.choice(n,1,p=normp)
        end        = sampleId[0]
        k          = end
        endGoal = self.goalTransitions[end]
        finishX, finishY, axis = uniform_sampling_1D(1, self.goalsData.goals_areas[endGoal,1:], self.goalsData.goals_areas[endGoal,0])

        # Use a pertubation approach to get the sample
        deltaX = finishX[0]-self.gpPathRegressor[k].finalAreaCenter[0]
        deltaY = finishY[0]-self.gpPathRegressor[k].finalAreaCenter[1]
        return self.gpPathRegressor[k].sample_path_with_perturbation(deltaX,deltaY)

    def sample_paths(self,nSamples):
        samples = []
        for k in range(nSamples):
            s = self.sample_path()
            samples.append(s)
        return samples
