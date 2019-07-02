from kernels import *
from statistics import*
from sampling import*
import matplotlib.pyplot as plt
import numpy as np
import math
from copy import copy

# This structure keeps all the learned data
# about the set of goals
class goalsLearnedStructure:

    # Constructor
    def __init__(self, areas, areasAxis):
        self.nGoals   = len(areas)
        self.areas    = areas
        self.areasAxis= areasAxis
        # Mean length for all pairs of goals
        self.meanLengths  = np.zeros((self.nGoals,self.nGoals))
        self.euclideanDistances = np.zeros((self.nGoals,self.nGoals))
        self.units    = np.zeros((self.nGoals,self.nGoals))
        self.meanUnit = 0.0
        self.priorTransitions = np.zeros((self.nGoals,self.nGoals))
        self.linearPriorsX = np.empty((self.nGoals, self.nGoals),dtype=object)
        self.linearPriorsY = np.empty((self.nGoals, self.nGoals),dtype=object)

    # Fills in the matrix with the
    # mean length of the trajectories
    def compute_mean_lengths(self,M):
        for i in range(self.nGoals):
            for j in range(self.nGoals):
                if(len(M[i][j]) > 0):
                    arclen = get_paths_arcLength(M[i][j])
                    m = np.median(arclen)
                else:
                    m = 0
                self.meanLengths[i][j] = m

    # Fills in the Euclidean distances between goals
    def compute_euclidean_distances(self):
        for i in range(self.nGoals):
            # Take the centroid of the ROI i
            p = middle_of_area(self.areas[i])
            for j in range(self.nGoals):
                # Take the centroid of the ROI j
                q = middle_of_area(self.areas[j])
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
