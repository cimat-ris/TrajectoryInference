"""
A class for GP-based path regression
"""
import numpy as np
import math
from gp_code.regression import *
from gp_code.likelihood import likelihood_from_partial_path
from gp_code.sampling import *
from utils.manip_trajectories import goal_center_and_size
from utils.manip_trajectories import euclidean_distance, positive_definite

class path_regression:
    # Constructor
    def __init__(self, kernelX, kernelY, unit, stepUnit, finalArea, finalAreaAxis,linearPriorX=None, linearPriorY=None, mode=None, timeData = None):
        self.mode            = mode # mode: Trautman or None
        self.observedX       = None
        self.observedY       = None
        self.observedL       = None
        self.speed           = None
        self.Kx              = None
        self.Ky              = None
        self.kx              = None
        self.ky              = None
        self.Cx              = None
        self.Cy              = None
        self.Kx_1            = None
        self.Ky_1            = None
        self.sqRootVarX      = np.empty((0, 0))
        self.sqRootVarY      = np.empty((0, 0))
        self.newL            = None
        self.epsilon         = 0.5
        self.kernelX         = kernelX
        self.kernelY         = kernelY
        self.linearPriorX    = linearPriorX
        self.linearPriorY    = linearPriorY
        self.unit            = unit
        self.stepUnit        = stepUnit
        self.finalArea       = finalArea
        self.finalAreaAxis   = finalAreaAxis
        self.finalAreaCenter, self.finalAreaSize = goal_center_and_size(finalArea)
        self.timeTransitionData = timeData # Data = {mean, std}

    # Update observations for the Gaussian process (matrix K)
    def updateObservations(self,observedX,observedY,observedL):
        # TODO: Kx,Ky could be simply updated instead of being redefined all the time
        n                    = len(observedX)
        self.observedX       = np.zeros((n+1,1))
        self.observedY       = np.zeros((n+1,1))
        self.observedL       = np.zeros((n+1,1))
        if self.mode == "Trautman" and n>1:
            dist = math.sqrt( (observedX[n-1] - observedX[n-2])**2 + (observedY[n-1] - observedY[n-2])**2 )
            self.speed =  dist/(observedL[n-1] - observedL[n-2]) #en este caso L es tiempo
        self.Kx              = np.zeros((n+1,n+1))
        self.Ky              = np.zeros((n+1,n+1))
        # Last really observed point
        lastObservedPoint = [observedX[-1], observedY[-1], observedL[-1]]
        # Generate the set of l values at which to predict x,y
        if self.mode == "Trautman":
            # Time difference between the last two observations
            timeStep    = (observedL[n-1] - observedL[n-2])
            # Elapsed time
            elapsedTime =  observedL[-1] - observedL[0]
            #print("\n*** Time Data *** \n",self.timeTransitionData)
            self.newL, finalL, self.dist = get_prediction_set_time(lastObservedPoint,elapsedTime,self.timeTransitionData,timeStep)

        else:
            # Determine the set of arclengths to predict
            self.newL, finalL, self.dist = get_prediction_set_arclengths(lastObservedPoint,self.finalAreaCenter,self.unit,self.stepUnit)
        # Fill in K (n+1 x n+1)
        for i in range(n):
            self.Kx[i][i] = self.kernelX(observedL[i],observedL[i])
            self.Ky[i][i] = self.kernelY(observedL[i],observedL[i])
            for j in range(i):
                self.Kx[i][j] = self.kernelX(observedL[i],observedL[j])
                self.Kx[j][i] = self.Kx[i][j]
                self.Ky[i][j] = self.kernelY(observedL[i],observedL[j])
                self.Ky[j][i] = self.Ky[i][j]
        # Last row/column
        for i in range(n):
            self.Kx[i][n] = self.kernelX(observedL[i],finalL)
            self.Kx[n][i] = self.Kx[i][n]
            self.Ky[i][n] = self.kernelY(observedL[i],finalL)
            self.Ky[n][i] = self.Ky[i][n]
        self.Kx[n][n] = self.kernelX(finalL,finalL)
        self.Ky[n][n] = self.kernelY(finalL,finalL)

        if self.mode == None:
            if self.finalAreaAxis==0:
                s              = self.finalAreaSize[0]
                self.Kx[n][n] += s*s*math.exp(-self.dist/s)
            if self.finalAreaAxis==1:
                s              = self.finalAreaSize[1]
                self.Ky[n][n] += s*s*math.exp(-self.dist/s)
        # Heavy
        self.Kx_1 = inv(self.Kx)
        self.Ky_1 = inv(self.Ky)
        for i in range(n):
            self.updateObserved(i,observedX[i],observedY[i],observedL[i])
        self.updateObserved(n,self.finalAreaCenter[0],self.finalAreaCenter[1],finalL)
        # For usage in prediction
        nnew         = len(self.newL)
        self.deltakx = np.zeros((nnew,1))
        self.deltaky = np.zeros((nnew,1))
        self.deltaKx = np.zeros((n+1,1))
        self.deltaKy = np.zeros((n+1,1))
        # Fill in deltakx
        for j in range(nnew):
            self.deltakx[j][0] = self.kernelX.dkdy(self.observedL[n],self.newL[j])
            self.deltaky[j][0] = self.kernelY.dkdy(self.observedL[n],self.newL[j])
        # Fill in deltaKx
        for j in range(n+1):
            self.deltaKx[j][0] = self.kernelX.dkdy(self.observedL[n],self.observedL[j])
            self.deltaKy[j][0] = self.kernelY.dkdy(self.observedL[n],self.observedL[j])

    # Update single observation i
    def updateObserved(self,i,x,y,l):
        # Center the data in case we use the linear prior
        if self.linearPriorX==None:
            self.observedX[i][0] = x
            self.observedY[i][0] = y
            self.observedL[i][0] = l
        else:
            self.observedX[i][0] = x - linear_mean(l, self.linearPriorX[0])
            self.observedY[i][0] = y - linear_mean(l, self.linearPriorY[0])
            self.observedL[i][0] = l

    # Compute the likelihood
    def computeLikelihood(self,observedX,observedY,observedL,startG,finishG,stepsToCompare,goalsData):
        # TODO: remove the goalsData structure
        self.likelihood = goalsData.priorTransitions[startG][finishG]*likelihood_from_partial_path(stepsToCompare,observedX,observedY,observedL,startG,finishG,goalsData)
        return self.likelihood

    # The main regression function: perform regression for a vector of values
    # lnew, that has been computed in update
    def prediction_to_finish_point(self):
        if self.newL==None:
            return None
        # Number of observed data
        n    = self.observedX.shape[0]
        # Number of predicted data
        nnew = len(self.newL)
        if nnew == 0:
            return None
        # Compute k (nxnnew), C (nnewxnnew)
        self.kx  = np.zeros((n,nnew))
        self.ky  = np.zeros((n,nnew))
        self.Cx  = np.zeros((nnew,nnew))
        self.Cy  = np.zeros((nnew,nnew))

        # Fill in k
        for i in range(n):
            for j in range(nnew):
                self.kx[i][j] = self.kernelX(self.observedL[i],self.newL[j],False)
                self.ky[i][j] = self.kernelY(self.observedL[i],self.newL[j],False)
        # Fill in C
        for i in range(nnew):
            for j in range(nnew):
                self.Cx[i][j] = self.kernelX(self.newL[i],self.newL[j],False)
                self.Cy[i][j] = self.kernelY(self.newL[i],self.newL[j],False)

        # Predictive mean
        self.Kx_1o= self.Kx_1.dot(self.observedX)
        self.Ky_1o= self.Ky_1.dot(self.observedY)
        self.newX = self.kx.transpose().dot(self.Kx_1o)
        self.newY = self.ky.transpose().dot(self.Ky_1o)
        if self.linearPriorX!=None:
            for j in range(nnew):
                self.newX[j] += linear_mean(self.newL[j],self.linearPriorX[0])
                self.newY[j] += linear_mean(self.newL[j],self.linearPriorY[0])
        # Estimate the variance in x
        self.ktKx_1 = self.kx.transpose().dot(self.Kx_1)
        kK_1kt      = self.ktKx_1.dot(self.kx)
        self.varx   = self.Cx - kK_1kt
        # Estimate the variance in y
        self.ktKy_1 = self.ky.transpose().dot(self.Ky_1)
        kK_1kt      = self.ktKy_1.dot(self.ky)
        self.vary   = self.Cy - kK_1kt
        # Regularization to avoid singular matrices
        self.varx += self.epsilon*np.eye(self.varx.shape[0])
        self.vary += self.epsilon*np.eye(self.vary.shape[0])
        # Cholesky on varX
        if positive_definite(self.varx) and positive_definite(self.vary):
            try:
                self.sqRootVarX = np.linalg.cholesky(self.varx)
            except np.linalg.LinAlgError:
                    self.varx = nearestPD(self.varx)
            try:
                self.sqRootVarY = np.linalg.cholesky(self.vary)
            except np.linalg.LinAlgError:
                self.vary = nearestPD(self.vary)
            self.sqRootVarX     = cholesky(self.varx,lower=True)
            self.sqRootVarY     = cholesky(self.vary,lower=True)
        return self.newX, self.newY, self.newL, self.varx, self.vary

    # Prediction as a perturbation of the "normal" prediction done to the center of an area
    def prediction_to_perturbed_finish_point(self,deltax,deltay):
        n            = len(self.observedX)
        nnew         = len(self.newL)
        # Express the displacement wrt the nominal ending point, as a nx1 vector
        deltaX       = np.zeros((n,1))
        deltaX[n-1,0]= deltax
        deltaY       = np.zeros((n,1))
        deltaY[n-1,0]= deltay
        # A first order approximation of the new final l
        deltal       = deltax*(self.finalAreaCenter[0]-self.observedX[n-2])/self.dist + deltay*(self.finalAreaCenter[1]-self.observedY[n-2])/self.dist
        # In this approximation, only the predictive mean is adapted (will be used for sampling)
        # First order term #1: variation in observedX
        newx = self.newX + self.ktKx_1.dot(deltaX)
        newy = self.newY + self.ktKy_1.dot(deltaY)
        # First order term #2: variation in kX
        newx+= self.Kx_1o[-1][0]*deltal*self.deltakx
        newy+= self.Ky_1o[-1][0]*deltal*self.deltaky
        # First order term #3: variation in Kx_1
        #  x= k^T (K+DK)^{-1}x
        #  x= k^T ((I+DK.K^{-1})K)^{-1}x
        #  x= k^T K^{-1}(I+DK.K^{-1})^{-1}x
        # dx=-k^T K^{-1}.DK.K^{-1}x
        newx-= deltal*self.Kx_1o[-1][0]*self.ktKx_1.dot(self.deltaKx)
        newx-= deltal*self.ktKx_1[0][-1]*self.deltaKx.transpose().dot(self.Kx_1o)
        newy-= deltal*self.Ky_1o[-1][0]*self.ktKy_1.dot(self.deltaKy)
        newy-= deltal*self.ktKy_1[0][-1]*self.deltaKy.transpose().dot(self.Ky_1o)
        return newx, newy, self.newL, self.varx, self.vary

    # Generate a sample from perturbations
    def sample_with_perturbation(self,deltaX,deltaY):
        # newx, newy, newl, varx, vary = self.prediction_to_finish_point() #we call this function to obtain newX, newY
        predictedX, predictedY, predictedL, varX, varY = self.prediction_to_perturbed_finish_point(deltaX,deltaY)
        # Number of predicted points
        nPredictions = len(predictedX)
        # Noise from a normal distribution
        sX = np.random.normal(size=(nPredictions,1))
        sY = np.random.normal(size=(nPredictions,1))
        if self.sqRootVarX.shape[0]>0 and self.sqRootVarY.shape[0]>0:
            return predictedX+self.sqRootVarX.dot(sX), predictedY+self.sqRootVarY.dot(sY), predictedL
        else:
            return predictedX, predictedY, predictedL

    # Generate a sample from the predictive distribution with a perturbed finish point
    def sample_with_perturbed_finish_point(self):
        # Sample end point around the sampled goal
        #finishX, finishY, axis = uniform_sampling_1D(1, self.finalAreaCenter, self.finalAreaAxis)
        size = self.finalAreaSize[self.finalAreaAxis]
        #print("\n***Final Area axis***\n", self.finalAreaAxis)
        finishX, finishY, axis = uniform_sampling_1D_around_point(1, self.finalAreaCenter,size, self.finalAreaAxis)
        # Use a pertubation approach to get the sample
        deltaX = finishX[0]-self.finalAreaCenter[0]
        deltaY = finishY[0]-self.finalAreaCenter[1]
        return self.sample_with_perturbation(deltaX,deltaY)