"""
Handling mixtures of GPs in trajectory prediction
"""
import numpy as np
import math
from gp_code.regression import *
from gp_code.sampling import *
from gp_code.trajectory_regression import *
from gp_code.trajectory_regressionT import *
from gp_code.likelihood import nearestPD
from utils.stats_trajectories import euclidean_distance
from utils.manip_trajectories import goal_center_and_size
from statistics import mean

# Class for performing trajectory regression with a mixture of Gaussian processes
class mGP_trajectory_prediction:

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
        # Starting goal
        self._start          = startG
        # Likelihoods
        self._goals_likelihood  = np.zeros(n, dtype=float)
        # Predicted means (per element of the mixture)
        self._predicted_means   = [np.zeros((0,3),  dtype=float)]*n
        self._predicted_vars    = [np.zeros((0,0,0),dtype=float)]*n
        self._filtered_paths    = [np.zeros((0,3),  dtype=float)]*n
        self._observed_x        = None
        self._observed_y        = None
        self._observed_l        = None
        self._current_time      = 0.0
        self._current_arc_length= 0.0
        self._speed_average     = 1.0

        # The basic elements here is this array of objects, that will do the regression work
        self.gpTrajectoryRegressor = [None]*n
        for i in range(self.goalsData.goals_n):
            if self.goalsData.kernelsX[self._start][i] is not None:
                timeTransitionData = None
                # If we have had enough data during the training phase to train the GP for this pa
                if self.goalsData.kernelsX[self._start][i].optimized:
                    # One regressor per goal
                    self.gpTrajectoryRegressor[i]=trajectory_regression(self.goalsData.kernelsX[self._start][i], self.goalsData.kernelsY[self._start][i],goalsData.sigmaNoise,self.goalsData.speedModels[self._start][i],self.goalsData.units[self._start][i],self.goalsData.goals_areas[i],prior=self.goalsData.priorTransitions[self._start][i], timeTransitionData=timeTransitionData)
                else:
                    k=self.goalsData.copyFromClosest(self._start,i)
                    self.gpTrajectoryRegressor[i]=trajectory_regression(self.goalsData.kernelsX[self._start][i], self.goalsData.kernelsY[self._start][i],goalsData.sigmaNoise,self.goalsData.speedModels[self._start][i],self.goalsData.units[self._start][k],self.goalsData.goals_areas[i],prior=self.goalsData.priorTransitions[self._start][i], timeTransitionData=timeTransitionData)

    # Update observations and compute likelihoods based on observations
    def update(self,observations,consecutiveObservations=True):
        if observations is None:
            return None
        nobs             = observations.shape[0]
        if nobs<2:
            return None
        self._observed_x = observations[:,0]
        self._observed_y = observations[:,1]
        self._observed_l = observations[:,2]
        # Update time and current arc length
        self._current_time      = observations[-1,3]
        self._current_arc_length= observations[-1,2]
        logging.debug('Time: {:2.2f} Arc-length: {:4.2f} Speed: {:2.2f}'.format(self._current_time,self._current_arc_length,self._speed_average))
        # Average speed
        weights = np.linspace(0.0,1.0,num=nobs)
        weights = weights/np.sum(weights)
        self._speed_average    = np.average(observations[:,4],axis=0,weights=weights)
        all_lk_predicted       = []
        # Update each regressor with its corresponding observations
        for i in range(self.goalsData.goals_n):
            if self.gpTrajectoryRegressor[i] is not None:
                logging.debug("Updating goal {:d}".format(i))
                goalCenter,__= goal_center_and_size(self.goalsData.goals_areas[i][1:])
                distToGoal   = euclidean_distance([self._observed_x[-1],self._observed_y[-1]], goalCenter)
                dist         = euclidean_distance([self._observed_x[0],self._observed_y[0]], goalCenter)
                # Update observations and re-compute the kernel matrices
                self.gpTrajectoryRegressor[i].update_observations(observations,consecutiveObservations)
                # Compute the model likelihood
                lkhd,lkhd_preds = self.gpTrajectoryRegressor[i].compute_likelihood()
                all_lk_predicted.append(lkhd_preds)
                self._goals_likelihood[i] =lkhd
            else:
                all_lk_predicted.append(None)
        # Sum of the likelihoods
        s = sum(self._goals_likelihood)
        # Compute the mean likelihood
        self.meanLikelihood = mean(self._goals_likelihood)
        # Maintain a list of likely goals
        for i in range(self.goalsData.goals_n):
            self._goals_likelihood[i] /= s
            if(self._goals_likelihood[i] > 0.85*self.meanLikelihood):
                self.likelyGoals.append(i)
        # Save most likely goal
        mostLikely = 0
        for i in range(self.goalsData.goals_n):
            if self._goals_likelihood[i] > self._goals_likelihood[mostLikely]:
                mostLikely = i
        self.mostLikelyGoal = mostLikely
        return self._goals_likelihood, all_lk_predicted

    # Get a filtered version of the initial observations
    def filter(self):
        # For all likely goals
        for i in range(self.goalsData.goals_n):
            if self.gpTrajectoryRegressor[i] is not None:
                self._filtered_paths[i]=self.gpTrajectoryRegressor[i].filter_observations()
        return self._filtered_paths

    # Performs path prediction
    def predict_path(self,compute_sqRoot=False):
        # For all likely goals
        for i in range(self.goalsData.goals_n):
            if self.gpTrajectoryRegressor[i] is not None:
                # Uses the already computed matrices to apply regression over missing data
                self._predicted_means[i], self._predicted_vars[i] = self.gpTrajectoryRegressor[i].predict_path_to_finish_point(compute_sqRoot=compute_sqRoot)
        return self._predicted_means,self._predicted_vars

    # Performs trajectory prediction
    def predict_trajectory(self,compute_sqRoot=False):
        # For all likely goals
        for i in range(self.goalsData.goals_n):
            if self.gpTrajectoryRegressor[i] is not None:
                # Uses the already computed matrices to apply regression over missing data
                self._predicted_means[i], self._predicted_vars[i] = self.gpTrajectoryRegressor[i].predict_trajectory_to_finish_point(self._speed_average,self._current_time,self._current_arc_length,compute_sqRoot=compute_sqRoot)
        return self._predicted_means,self._predicted_vars

    def sample_path(self,efficient=True):
        p = self._goals_likelihood[:self.goalsData.goals_n]
        # Sample goal (discrete choice)
        goalSample = np.random.choice(self.goalsData.goals_n,1,p=p)
        end        = goalSample[0]
        k          = end
        finishX, finishY, axis = uniform_sampling_1D(1, self.goalsData.goals_areas[end][1:], self.goalsData.goals_areas[end][0])
        # Use a pertubation approach to get the sample
        deltaX = finishX[0]-self.gpTrajectoryRegressor[k].finalAreaCenter[0]
        deltaY = finishY[0]-self.gpTrajectoryRegressor[k].finalAreaCenter[1]
        return self.gpTrajectoryRegressor[k].sample_path_with_perturbation(deltaX,deltaY,efficient),[finishX, finishY]

    # Generate samples from the predictive distribution
    def sample_paths(self,nSamples,efficient=True):
        vec = []
        for k in range(nSamples):
            path, finalPosition = self.sample_path(efficient)
            vec.append(path)
        return vec
