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
        # Max number of elements in the mixture (not all are used at the same time)
        maxn = 10
        # Array of potential future goals
        arr = np.random.choice([i for i in range(goalsData.nGoals)], maxn, replace=False, p=goalsData.priorTransitions[startG])
        # Select elements where timeTransition is not zero
        deleteid = []
        for i in range(maxn):
            if goalsData.timeTransitionMeans[startG][arr[i]] == 0:
                deleteid.append(i)
        self.goalTransitions = np.delete(arr, deleteid)
        n                    = self.goalTransitions.size
        # Points to evaluate the likelihoods
        self.nPoints         = 5
        # Step unit
        self.stepUnit        = goalsData.stepUnit # ---> time unit?
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
        self.observedT       = None
        # The basic element here is this object, that will do the regression work
        self.gpPathRegressor = [None]*n
        self.gpTrajectoryRegressor = [None]*n
        
        for i in range(n):
            gi = self. goalTransitions[i]
            timeTransitionData = [self.goalsData.timeTransitionMeans[self.startG][gi],self.goalsData.timeTransitionStd[self.startG][gi]]
            self.gpPathRegressor[i] = path_regression(self.goalsData.kernelsX[self.startG][gi], self.goalsData.kernelsY[self.startG][gi],None,None,self.goalsData.areas_coordinates[gi],self.goalsData.areas_axis[gi],None,'Trautman',timeTransitionData)
    
    def update(self, observations):
        self.observedX = observations[:,0]
        self.observedY = observations[:,1]
        self.observedT = observations[:,2]
        # Update each regressor with its corresponding observations
        for i in range(len(self.goalTransitions)):
            # Update observations and re-compute the kernel matrices
            self.gpPathRegressor[i].update_observations(observations)
            # Compute the model likelihood
            self.goalsLikelihood[i] = self.gpPathRegressor[i].compute_likelihood(observations,self.nPoints)

        # Compute the mean likelihood
        self.meanLikelihood = mean(self.goalsLikelihood)
        
        n = len(self.goalTransitions)
        mostLikely = 0
        for i in range(n):
            if self.goalsLikelihood[i] > self.goalsLikelihood[mostLikely]:
                mostLikely = i
        self.mostLikelyGoal = mostLikely

        return self.goalsLikelihood

    # Performs prediction
    def predict_path(self):
        n = len(self.goalTransitions)
        # For all likely goals
        for i in range(n):#self.goalsData.nGoals):
            gi = self.goalTransitions[i]
            print('[INF] Predicting to goal ',gi)
            goalCenter,__ = goal_center_and_size(self.goalsData.areas_coordinates[gi])
            #distToGoal    = euclidean_distance([self.observedX[-1],self.observedY[-1]], goalCenter)
            #dist          = euclidean_distance([self.observedX[0],self.observedY[0]], goalCenter)

            # Uses the already computed matrices to apply regression over missing data
            self.predictedMeans[i], self.predictedVars[i] = self.gpPathRegressor[i].predict_path_to_finish_point()
            
        return self.predictedMeans,self.predictedVars
    
    def sample_path(self):
        n = len(self.goalTransitions)
        p = self.goalsLikelihood[:n]
        #TODO: check probabilities
        
        # Sample goal
        sampleId = np.random.choice(n,1,p=p)
        end        = sampleId[0]
        k          = end
        
        endGoal = self.goalTransitions[end]
        finishX, finishY, axis = uniform_sampling_1D(1, self.goalsData.areas_coordinates[endGoal], self.goalsData.areas_axis[endGoal])
        
        # Use a pertubation approach to get the sample
        deltaX = finishX[0]-self.gpPathRegressor[k].finalAreaCenter[0]
        deltaY = finishY[0]-self.gpPathRegressor[k].finalAreaCenter[1]
        return self.gpPathRegressor[k].sample_path_with_perturbation(deltaX,deltaY)

    def sample_paths(self,nSamples):
        vecX, vecY, vecT = [], [], []
        for k in range(nSamples):
            x, y, t = self.sample_path()
            vecX.append(x)
            vecY.append(y)
            vecL.append(t)
        return vecX,vecY,vecT
        