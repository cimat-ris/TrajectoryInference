from gp_code.kernels import set_kernel
from gp_code.optimize_parameters import *
from utils.stats_trajectories import trajectory_arclength, trajectory_duration
from utils.manip_trajectories import get_linear_prior_mean
from utils.manip_trajectories import get_data_from_set
from utils.manip_trajectories import goal_center
from utils.stats_trajectories import euclidean_distance, avg_speed, median_speed
from utils.stats_trajectories import truncate
from sklearn.linear_model import LinearRegression, HuberRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import logging
import numpy as np

# This structure keeps all the learned data
# about the set of goals
class goal_pairs:

    # Constructor
    def __init__(self, goals_areas, trajMat, sigmaNoise=85.0, min_traj_number=5):
        self.goals_n            = len(goals_areas)
        self.goals_areas        = goals_areas
        self.min_traj_number    = min_traj_number
        # Observation Noise
        self.sigmaNoise         = sigmaNoise
        # Minimum value for speed to avoid numerical problems
        self.epsilon            = 0.1
        # Flag to know if the pairs of goals have parameters
        # that have been learned
        self.learned        = np.zeros((self.goals_n,self.goals_n),dtype=int)
        # Mean length for all pairs of goals
        self.meanLengths        = np.zeros((self.goals_n,self.goals_n))
        self.euclideanDistances = np.zeros((self.goals_n,self.goals_n))
        self.units              = np.zeros((self.goals_n,self.goals_n))
        self.priorTransitions   = np.zeros((self.goals_n,self.goals_n))
        self.kernelsX           = np.empty((self.goals_n,self.goals_n),dtype=object)
        self.kernelsY           = np.empty((self.goals_n,self.goals_n),dtype=object)
        self.speedModels        = np.zeros((self.goals_n,self.goals_n),dtype=object)
        self.timeTransitionMeans= np.empty((self.goals_n,self.goals_n),dtype=object)
        self.timeTransitionStd  = np.empty((self.goals_n,self.goals_n),dtype=object)
        # Compute the mean lengths
        self.compute_mean_lengths(trajMat)
        # Compute the distances between pairs of goals (as a nGoalsxnGoals matrix)
        self.compute_euclidean_distances()
        # Compute the ratios between average path lengths and inter-goal distances
        self.compute_distance_unit()
        # Computer prior probabispeedRegressorlities between goals
        self.compute_prior_transitions(trajMat)
        # Compute transition probabilities between goals
        self.compute_time_transitions(trajMat)
        # Compute speed models
        self.optimize_speed_models(trajMat)

    # Fills in the matrix with the mean length of the trajectories
    def compute_mean_lengths(self, trajectories):
        # For each pair of goals (gi,gj), compute the mean arc-length
        for i in range(self.goals_n):
            for j in range(self.goals_n):
                if len(trajectories[i][j]) > 0:
                    arclengths = []
                    for trajectory in trajectories[i][j]:
                        tr_arclen = trajectory_arclength(trajectory)
                        arclengths.append(tr_arclen[-1])
                    m = np.mean(arclengths)
                else:
                    m = 0
                self.meanLengths[i][j] = m

    # Fills in the Euclidean distances between goals
    def compute_euclidean_distances(self):
        for i in range(self.goals_n):
            # Take the centroid of the ROI i
            p = goal_center(self.goals_areas[i][1:])
            for j in range(self.goals_n):
                # Take the centroid of the ROI j
                q = goal_center(self.goals_areas[j][1:])
                d = euclidean_distance(p,q)
                self.euclideanDistances[i][j] = d

    # Computes the ratio between path length and linear path length
    # (distance between goals)
    def compute_distance_unit(self):
        for i in range(self.goals_n):
            for j in range(self.goals_n):
                if(self.euclideanDistances[i][j] == 0 or self.meanLengths[i][j] == 0):
                    u = 1.0
                else:
                    # Ratio between mean length and goal-to-goal distance
                    u = self.meanLengths[i][j]/self.euclideanDistances[i][j]
                self.units[i][j] = u

    # Fills in the probability transition matrix gi -> gj
    def compute_prior_transitions(self,pathMat):
        for i in range(self.goals_n):
            count = 0.
            for j in range(self.goals_n):
                count += len(pathMat[i][j])
            for j in range(self.goals_n):
                if count == 0:
                    self.priorTransitions[i][j] = 0.
                else:
                    val = float(len(pathMat[i][j])/count)
                    self.priorTransitions[i][j] = truncate(val,8)
            s = np.sum(self.priorTransitions[i])
            if s > 0.0 and s < 1.0:
                d = truncate(1.0 - s,8)
                self.priorTransitions[i][i] += float(d)

    # For each pair, optimize speed model
    def optimize_speed_models(self,trainingSet):
        # For all the trajectories
        for i in range(self.goals_n):
            for j in range(self.goals_n):
                trajSet        = trainingSet[i][j]
                # Only if we have enough trajectories
                if len(trajSet) < self.min_traj_number:
                    continue
                relativeSpeeds = []
                lengths        = []
                for tr in trajSet:
                    # Times
                    t = tr[2]
                    # Average speed
                    v = avg_speed(tr)+self.epsilon
                    # Arc lengths
                    d = trajectory_arclength(tr)
                    for k in range(1,len(t)):
                        relativeSpeeds.append(float((d[k]-d[k-1])/(t[k]-t[k-1]))/v)
                        lengths.append(d[k])
                lengths        = np.array(lengths).reshape(-1, 1)
                relativeSpeeds = np.array(relativeSpeeds)
                self.speedModels[i][j]=make_pipeline(PolynomialFeatures(4),LinearRegression())
                self.speedModels[i][j].fit(lengths, relativeSpeeds)


    # For each pair of goals, realize the optimization of the kernel parameters
    def optimize_kernel_parameters(self,kernelType,trainingSet):
        # Build the kernel matrices with the default values
        self.kernelsX = create_kernel_matrix(kernelType, self.goals_n, self.goals_n)
        self.kernelsY = create_kernel_matrix(kernelType, self.goals_n, self.goals_n)
        logging.info("Optimizing kernel parameters")
        # For every pair of goals (gi, gj)
        for i in range(self.goals_n):
            for j in range(self.goals_n):
                # Get the set of paths that go from i to j
                paths = trainingSet[i][j]
                # We define a GP only if we have enough trajectories
                if len(paths) > self.min_traj_number:
                    start = timeit.default_timer()
                    # Get the path data as x,y,z (z is arclength)
                    x,y,l = get_data_from_set(paths)
                    # Build a kernel with the specified type and initial parameters theta
                    ker   = set_kernel(kernelType)
                    # Set the linear prior
                    if self.kernelsX[i][j].linearPrior:
                        meanX, varX  = get_linear_prior_mean(trainingSet[i][j], 'x')
                        ker.set_linear_prior(meanX[0],meanX[1],varX[0],varX[1])
                    theta  = ker.get_optimizable_parameters()
                    logging.info("[{:d}][{:d}]".format(i,j))
                    logging.info("#trajectories: {:d}".format(len(l)))
                    logging.info("Initial values for the optimizable parameters: {}".format(theta))
                    # Fit parameters in X
                    thetaX  = fit_parameters(l,x,ker,theta,self.sigmaNoise)
                    logging.info("Optimized parameters for x: {}".format(thetaX))
                    self.kernelsX[i][j].set_parameters(ker.get_parameters())
                    logging.info("Full parameters for x: {}".format(self.kernelsX[i][j].get_parameters()))
                    # Fit parameters in Y
                    ker   = set_kernel(kernelType)
                    if self.kernelsY[i][j].linearPrior:
                        meanY, varY  = get_linear_prior_mean(trainingSet[i][j], 'y')
                        ker.set_linear_prior(meanY[0],meanY[1],varY[0],varY[1])
                    thetaY  = fit_parameters(l,y,ker,theta,self.sigmaNoise)
                    logging.info("Optimized parameters for y: {}",thetaY)
                    self.kernelsY[i][j].set_parameters(ker.get_parameters())
                    logging.info("Full parameters for y: {}",self.kernelsY[i][j].get_parameters())
                    stop = timeit.default_timer()
                    execution_time = stop - start
                    logging.info("Parameter optimization done in {:2f} seconds".format(execution_time))
                else:
                    self.kernelsX[i][j] = None
                    self.kernelsY[i][j] = None

    # Fills in the probability transition matrix
    def compute_time_transitions(self, trMat):
        for i in range(self.goals_n):
            for j in range(self.goals_n):
                M, SD = 0, 0
                duration = []
                if(len(trMat[i][j]) > self.min_traj_number):
                    for tr in trMat[i][j]:
                        duration.append(trajectory_duration(tr))
                    M  = np.mean(duration)
                    SD = np.std(duration)

                self.timeTransitionMeans[i][j] = M
                self.timeTransitionStd[i][j]   = SD
