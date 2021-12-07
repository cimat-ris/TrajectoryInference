"""
A class for GP-based trajectory regression (path AND time)
"""
import numpy as np
import math
import logging
from gp_code.path_regression import *
from utils.stats_trajectories import avg_speed


"""
trajectory_regression inherits from path_regression
"""
class trajectory_regression(path_regression):

    # Constructor
    def __init__(self, kernelX, kernelY, sigmaNoise, speedModel, unit, finalArea, prior=0.0, timeTransitionData=None):
        # Init of the base class
        super(trajectory_regression, self).__init__(kernelX, kernelY, sigmaNoise, unit, finalArea, prior, timeTransitionData)
        # Regression model for the relative value of speed
        self.speedModel       = speedModel

    # Prediction of the trajectory
    def predict_trajectory_to_finish_point(self,current_speed,current_time,current_arc_length,compute_sqRoot=False):
        # Predict path
        path, var   = self.predict_path_to_finish_point(compute_sqRoot)
        if path is None:
            return None,None
        arc_lengths     = path[:,2]
        arc_lengths_ext = np.concatenate([[current_arc_length],path[:,2]])
        # Use the speed model to predict relative speed with respect to the average
        if isinstance(self.speedModel, int):
            predicted_relative_speeds = np.ones(arc_lengths.shape)
        else:
            predicted_relative_speeds = self.speedModel.predict(arc_lengths.reshape(-1, 1))
        predicted_speeds          = current_speed*predicted_relative_speeds.reshape(-1, 1)
        times = current_time + np.cumsum(np.divide((arc_lengths_ext[1:]-arc_lengths_ext[:-1]).reshape(-1,1),predicted_speeds)).reshape(-1,1)
        trajectory = np.concatenate([path,times,predicted_speeds],axis=1)
        return trajectory, var

    # Update observations
    def update_observations(self,observations,consecutiveObservations=True):
        # Call the method for the path regression
        super(trajectory_regression, self).update_observations(observations,consecutiveObservations)


    # Generate a sample from perturbations
    def sample_trajectory_with_perturbation(self,deltaX,deltaY):
        pass

    # Generate a sample from the predictive distribution with a perturbed finish point
    def sample_trajectory_with_perturbed_finish_point(self):
        pass
