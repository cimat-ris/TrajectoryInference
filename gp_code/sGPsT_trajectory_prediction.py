"""
A class for handling a single GP in trajectory prediction
"""
import numpy as np
import math, sys, os
import torch
from gp_code.sGP_trajectory_prediction import *
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../trajnetplusplustools')))
from utils.manip_trajectories import start_time, end_time, get_trajectories_given_time_interval, get_goal_of_point
#import trajnetplusplustools
from trajnetplusplusbaselines.trajnetbaselines.lstm import LSTMPredictor
from trajnetplusplustools.reader import Reader
from trajnetplusplustools.data import TrackRow

# Class for performing path regression with a single Gaussian Process
class sGPsT_trajectory_prediction(sGP_trajectory_prediction):

    # Constructor
    def __init__(self, startG, endG, goalsData, mode = None):
        # Init of the base class
        super(sGPsT_trajectory_prediction, self).__init__(startG, endG, goalsData, mode)
        # Instanciate a predictor
        predictor = LSTMPredictor.load("parameters/sgan_directional_None.pkl")
        # On CPU
        device    = torch.device('cpu')
        predictor.model.to(device)

    # Update observations and compute likelihood based on observations
    def update(self,observations,consecutiveObservations=True):
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
        self.gpTrajectoryRegressor.update_observations(observations,consecutiveObservations)
        # Compute the model likelihood
        return self.gpTrajectoryRegressor.compute_likelihood()
