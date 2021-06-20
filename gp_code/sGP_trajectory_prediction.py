"""
A class for handling a single GP in trajectory prediction
"""
import numpy as np
import math
from gp_code.trajectory_regression import *

# Class for performing path regression with a single Gaussian Process
class sGP_trajectory_prediction:

    # Constructor
    def __init__(self, startG, endG, goalsData, mode = None):
        self.goalsData       = goalsData
        self.nPoints         = 5
        self.startG          = startG
        self.endG            = endG
        self.predictedMeans  = None
        self.predictedVars   = None
        self._current_time      = 0.0
        self._current_arc_length= 0.0
        self._speed_average     = 1.0

        # The basic element here is this object, that will do the regression work
        if mode == 'Trautman': #!Using path_regression for now
            timeTransitionData = [self.goalsData.timeTransitionMeans[self.startG][self.endG],self.goalsData.timeTransitionStd[self.startG][self.endG]]
            self.gpTrajectoryRegressor = path_regression(self.goalsData.kernelsX[self.startG][self.endG], self.goalsData.kernelsY[self.startG][self.endG],goalsData.sigmaNoise,self.goalsData.units[self.startG][self.endG],self.goalsData.goals_areas[self.endG],mode='Trautman',timeTransitionData=timeTransitionData)
        else:
            self.gpTrajectoryRegressor = trajectory_regression(self.goalsData.kernelsX[self.startG][self.endG], self.goalsData.kernelsY[self.startG][self.endG],goalsData.sigmaNoise,self.goalsData.speedModels[self.startG][self.endG],self.goalsData.units[self.startG][self.endG],self.goalsData.goals_areas[self.endG],self.goalsData.priorTransitions[self.startG][self.endG])

    # Update observations and compute likelihood based on observations
    def update(self,observations):
        # Update time and current arc length
        self._current_time      = observations[-1,3]
        self._current_arc_length= observations[-1,2]
        logging.info('Time: {:2.2f} Arc-length: {:4.2f} Speed: {:2.2f}'.format(self._current_time,self._current_arc_length,self._speed_average))
        # Average speed
        nobs    = observations.shape[0]
        weights = np.linspace(0.0,1.0,num=nobs)
        weights = weights/np.sum(weights)
        self._speed_average    = np.average(observations[:,4],axis=0,weights=weights)
        # Update observations and re-compute the kernel matrices
        self.gpTrajectoryRegressor.update_observations(observations)
        # Compute the model likelihood
        return self.gpTrajectoryRegressor.compute_likelihood()

    # Performs path prediction
    def predict_path(self,compute_sqRoot=False):
        # Uses the already computed matrices to apply regression over missing data
        self.predictedMeans, self.predictedVars = self.gpTrajectoryRegressor.predict_path_to_finish_point(compute_sqRoot=compute_sqRoot)
        return self.predictedMeans,self.predictedVars

    # Performs trajectory prediction
    def predict_trajectory(self,compute_sqRoot=False):
        # Uses the already computed matrices to apply regression over missing data
        self.predictedMeans, self.predictedVars = self.gpTrajectoryRegressor.predict_trajectory_to_finish_point(self._speed_average,self._current_time,self._current_arc_length,compute_sqRoot=compute_sqRoot)
        return self.predictedMeans,self.predictedVars

    # Get a filtered version of the initial observations
    def filter(self):
        return self.gpTrajectoryRegressor.filter_observations()

    # Generate a sample from the current Gaussian predictive distribution
    def sample_path(self):
        return self.gpTrajectoryRegressor.sample_path_with_perturbed_finish_point()

    # Generate samples from the predictive distribution
    def sample_paths(self,nSamples):
        vec = []
        for k in range(nSamples):
            path = self.sample_path()
            vec.append(path)
        return vec
