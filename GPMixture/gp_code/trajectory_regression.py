"""
A class for GP-based trajectory regression (path AND time)
"""
import numpy as np
import math
from gp_code.path_regression import *
from utils.stats_trajectories import avg_speed


class trajectory_regression(path_regression):

    # Constructor
    def __init__(self, kernelX, kernelY, speedModel,unit, stepUnit, finalArea, finalAreaAxis, prior):
        # Init of the base class
        super(trajectory_regression, self).__init__(kernelX, kernelY, unit, stepUnit, finalArea, finalAreaAxis, prior)
        # Regression model for the relative value of speed
        self.speedModel  = speedModel
        self.speedAverage= 1.0

    def predict_trajectory_to_finish_point(self,compute_sqRoot=False):
        # Predict path
        path, var   = self.predict_path_to_finish_point(compute_sqRoot)
        arc_lengths = path[:,2]
        # Use the speed model to predict relative speed with respect to the average
        if isinstance(self.speedModel, int):
            predicted_relative_speeds = 1.0
        else:
            predicted_relative_speeds = self.speedModel.predict(arc_lengths.reshape(-1, 1))
        predicted_speeds          = self.speedAverage*predicted_relative_speeds
        return path, var

    # Update observations
    def update_observations(self,observations):
        # Call the method for the path regression
        super(trajectory_regression, self).update_observations(observations)
        # Average speed
        self.speedAverage = np.mean(observations[:,3],axis=0)

    # Generate a sample from perturbations
    def sample_trajectory_with_perturbation(self,deltaX,deltaY):
        pass

    # Generate a sample from the predictive distribution with a perturbed finish point
    def sample_trajectory_with_perturbed_finish_point(self):
        pass
