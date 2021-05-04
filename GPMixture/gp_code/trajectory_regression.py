"""
A class for GP-based trajectory regression (path AND time)
"""
import numpy as np
import math
from gp_code.path_regression import *
from utils.stats_trajectories import avg_speed


class trajectory_regression(path_regression):

    # Constructor
    def __init__(self, kernelX, kernelY, sigmaNoise, speedModel, unit, finalArea, prior=0.0):
        # Init of the base class
        super(trajectory_regression, self).__init__(kernelX, kernelY, sigmaNoise, unit, finalArea, prior)
        # Regression model for the relative value of speed
        self.speedModel       = speedModel
        self.speedAverage     = 1.0
        self.currentTime      = 0.0
        self.currentArcLength = 0.0

    # Prediction of the trajectory
    def predict_trajectory_to_finish_point(self,compute_sqRoot=False):
        # Predict path
        path, var   = self.predict_path_to_finish_point(compute_sqRoot)
        arc_lengths     = path[:,2]
        arc_lengths_ext = np.concatenate([[self.currentArcLength],path[:,2]])
        # Use the speed model to predict relative speed with respect to the average
        if isinstance(self.speedModel, int):
            predicted_relative_speeds = np.ones(arc_lengths.shape)
        else:
            predicted_relative_speeds = self.speedModel.predict(arc_lengths.reshape(-1, 1))
        predicted_speeds          = self.speedAverage*predicted_relative_speeds.reshape(-1, 1)
        times = self.currentTime + np.cumsum(np.divide((arc_lengths_ext[1:]-arc_lengths_ext[:-1]).reshape(-1,1),predicted_speeds)).reshape(-1,1)
        trajectory = np.concatenate([path,times,predicted_speeds],axis=1)
        return trajectory, var

    # Update observations
    def update_observations(self,observations):
        # Call the method for the path regression
        super(trajectory_regression, self).update_observations(observations)
        # Average speed
        nobs    = observations.shape[0]
        weights = np.linspace(0.0,1.0,num=nobs)
        weights = weights/np.sum(weights)
        self.speedAverage = np.average(observations[:,4],axis=0,weights=weights)
        print('[INF] Speed average: ',self.speedAverage)
        self.currentTime = observations[-1,3]
        print('[INF] Current time: ',self.currentTime)
        self.currentArcLength = observations[-1,2]
        print('[INF] Current arc-length: ',self.currentArcLength)

    # Generate a sample from perturbations
    def sample_trajectory_with_perturbation(self,deltaX,deltaY):
        pass

    # Generate a sample from the predictive distribution with a perturbed finish point
    def sample_trajectory_with_perturbed_finish_point(self):
        pass
