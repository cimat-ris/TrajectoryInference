"""
Handling mixtures of GPs in trajectory prediction | Trautman
"""
import numpy as np
from statistics import mean
from gp_code.sampling import *
from gp_code.path_regression import path_regression
from gp_code.trajectory_regression import trajectory_regression
from gp_code.likelihood import nearestPD
from utils.stats_trajectories import euclidean_distance
from utils.manip_trajectories import goal_center_and_size

# Class for performing path regression with a mixture of Gaussian processes | Trautman's approach
class mixtureGPT:
    #TODO: add flag to path regressor for Trautman's mode
    def __init__(self, startG, goalsData):
        # The goals structure
        self.goalsData       = goalsData
        # Sub-set of likely goals
        self.likelyGoals     = []
        #Index of most likely goal
        self.mostLikelyGoal = None
        # Number of elements in the mixture (not all are used at the same time)
        n                    = self.goalsData.nGoals
        # Points to evaluate the likelihoods
        self.nPoints         = 5
        # Step unit
        self.stepUnit        = goalsData.stepUnit # ---> time unit
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
            self.gpPathRegressor[i] = path_regression(self.goalsData.kernelsX[self.startG][i], self.goalsData.kernelsY[self.startG][i],goalsData.distUnit,self.stepUnit,self.goalsData.areas_coordinates[i],self.goalsData.areas_axis[i])
            #self.gpTrajectoryRegressor[i] = trajectory_regression(self.goalsData.kernelsX[self.startG][i], self.goalsData.kernelsY[self.startG][i],goalsData.distUnit,stepUnit,self.goalsData.areas_coordinates[i],self.goalsData.areas_axis[i])
