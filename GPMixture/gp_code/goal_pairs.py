from gp_code.kernels import set_kernel
from gp_code.optimize_parameters import *
from utils.stats_trajectories import get_paths_arclength, get_paths_duration
from utils.manip_trajectories import goal_center_and_size, get_linear_prior_mean
from utils.stats_trajectories import get_data_from_paths
import numpy as np
import math
from copy import copy

# This structure keeps all the learned data
# about the set of goals
class goal_pairs:

    # Constructor
    def __init__(self, areas_coordinates, areas_axis, trajectories, min_traj_number=15):
        self.nGoals                = len(areas_coordinates)
        self.areas_coordinates     = areas_coordinates
        self.areas_axis            = areas_axis
        self.min_traj_number       = min_traj_number
        # Mean length for all pairs of goals
        self.meanLengths        = np.zeros((self.nGoals,self.nGoals))
        self.euclideanDistances = np.zeros((self.nGoals,self.nGoals))
        self.units              = np.zeros((self.nGoals,self.nGoals))
        self.meanUnit           = 0.0
        self.priorTransitions   = np.zeros((self.nGoals,self.nGoals))
        #self.linearPriorsX      = np.empty((self.nGoals, self.nGoals),dtype=object)
        #self.linearPriorsY      = np.empty((self.nGoals, self.nGoals),dtype=object)
        self.kernelsX           = np.empty((self.nGoals, self.nGoals),dtype=object)
        self.kernelsY           = np.empty((self.nGoals, self.nGoals),dtype=object)
        self.timeTransitionMeans= np.empty((self.nGoals, self.nGoals),dtype=object)
        self.timeTransitionStd  = np.empty((self.nGoals, self.nGoals),dtype=object)
        # Compute the mean lengths
        self.compute_mean_lengths(trajectories)
        # Compute the distances between pairs of goals (as a nGoalsxnGoals matrix)
        self.compute_euclidean_distances()
        # Compute the ratios between average path lengths and inter-goal distances
        self.compute_distance_units()
        # Computer prior probabispeedRegressorlities between goals
        self.compute_prior_transitions(trajectories)
        # Compute transition probabilities between goals
        self.compute_time_transitions(trajectories)

    # Fills in the matrix with the
    # mean length of the trajectories
    def compute_mean_lengths(self,M):
        for i in range(self.nGoals):
            for j in range(self.nGoals):
                if(len(M[i][j]) > 0):
                    arclen = get_paths_arclength(M[i][j])
                    m = np.median(arclen)
                else:
                    m = 0
                self.meanLengths[i][j] = m

    # Fills in the Euclidean distances between goals
    def compute_euclidean_distances(self):
        for i in range(self.nGoals):
            # Take the centroid of the ROI i
            p,__ = goal_center_and_size(self.areas_coordinates[i])
            for j in range(self.nGoals):
                # Take the centroid of the ROI j
                q,__ = goal_center_and_size(self.areas_coordinates[j])
                # Compute the euclidean distance between the two centroids i and j
                d = np.sqrt((p[0]-q[0])**2 + (p[1]-q[1])**2)
                self.euclideanDistances[i][j] = d

    # Determine the ratio between path length and linear path length (distance between goals)
    def compute_distance_units(self):
        for i in range(self.nGoals):
            for j in range(self.nGoals):
                if(self.euclideanDistances[i][j] == 0 or self.meanLengths[i][j] == 0):
                    u = 1
                else:
                    # Ratio between mean length and goal-to-goal distance
                    u = self.meanLengths[i][j]/self.euclideanDistances[i][j]
                self.units[i][j] = u

        # Mean value of the ratio
        sumUnit = 0.
        for i in range(self.nGoals):
            for j in range(self.nGoals):
                sumUnit += self.units[i][j]
        self.meanUnit = sumUnit / self.nGoals**2

    # Fills in the probability transition matrix
    def compute_prior_transitions(self,pathMat):
        for i in range(self.nGoals):
            paths_i = 0.
            for j in range(self.nGoals):
                paths_i += len(pathMat[i][j])
            for j in range(self.nGoals):
                if paths_i == 0:
                    self.priorTransitions[i][j] = 0.
                else:
                    self.priorTransitions[i][j] = float(len(pathMat[i][j])/paths_i)

    # Compute, for X and Y, for each pair of goals, the matrix of the linear prior means
    def compute_linear_priors(self,pathMat):
        for i in range(self.nGoals):
            for j in range(self.nGoals):
                meanX, covX, varX  = get_linear_prior_mean(pathMat[i][j], 'x')
                meanY, covY, varY  = get_linear_prior_mean(pathMat[i][j], 'y')
                self.linearPriorsX[i][j] = (meanX,varX)
                self.linearPriorsY[i][j] = (meanY,varY)

    # For each pair of goals, realize the optimization of the kernel parameters
    def optimize_kernel_parameters(self,kernelType,trainingSet):
        # Build the kernel matrices with the default values
        self.kernelsX, parametersX = create_kernel_matrix(kernelType, self.nGoals, self.nGoals)
        self.kernelsY, parametersY = create_kernel_matrix(kernelType, self.nGoals, self.nGoals)
        # For goal i
        for i in range(self.nGoals):
            # For goal j
            for j in range(self.nGoals):
                # Get the set of paths that go from i to j
                paths = trainingSet[i][j]
                # We define a GP only if we have enough trajectories
                if len(paths) > self.min_traj_number:
                    start = timeit.default_timer()
                    # Get the path data as x,y,z (z is arclength)
                    x,y,__,l,__ = get_data_from_paths(paths)
                    # Build a kernel with the specified type and initial parameters theta
                    ker   = set_kernel(kernelType)
                    params= ker.get_parameters()
                    theta = ker.get_optimizable_parameters()
                    print("[OPT] Init parameters ",theta)
                    print("[OPT] [",i,"][",j,"]")
                    print("[OPT] #trajectories: ",len(l))
                    # Fit parameters in X
                    thetaX  = fit_parameters(l,x,ker,theta)
                    print("[OPT] x: ",thetaX)
                    self.kernelsX[i][j].set_parameters(ker.get_parameters())
                    # Fit parameters in Y
                    thetaY  = fit_parameters(l,y,ker,theta)
                    print("[OPT] y: ",thetaY)
                    self.kernelsY[i][j].set_parameters(ker.get_parameters())
                    stop = timeit.default_timer()
                    execution_time = stop - start
                    print("[OPT] Parameter optimization done in %.2f seconds"%execution_time)
                    # Evaluate steps_unit
                    # self.stepUnit = get_steps_unit(paths)
                else:
                    self.kernelsX[i][j] = None
                    self.kernelsY[i][j] = None

    # Fills in the probability transition matrix
    def compute_time_transitions(self,pathMat):
        for i in range(self.nGoals):
            for j in range(self.nGoals):
                m, std = 0,0
                if(len(pathMat[i][j]) > self.min_traj_number):
                    time = get_paths_duration(pathMat[i][j])
                    m   = np.median(time)
                    std = np.std(time)
                self.timeTransitionMeans[i][j] = m
                self.timeTransitionStd[i][j]   = std
