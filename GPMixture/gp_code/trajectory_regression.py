"""
A class for GP-based speed regression
"""
import numpy as np
import math
from gp_code.path_regression import *
from gp_code.speed_regression import *


# TODO: Reintegrate this stuff here as part of trajectory regression
#        if self.mode == "Trautman" and n>1:
#            dist = math.sqrt( (observedX[n-1] - observedX[n-2])**2 + (observedY[n-1] - observedY[n-2])**2 )
#            self.speed =  dist/(observedL[n-1] - observedL[n-2]) #en este caso L es tiempo

# Generate the set of l values at which to predict x,y
# if self.mode == "Trautman":
# Time difference between the last two observations
# timeStep    = (observedL[n-1] - observedL[n-2])
# Elapsed time
# elapsedTime =  observedL[-1] - observedL[0]
# print("\n*** Time Data *** \n",self.timeTransitionData)
# self.newL, finalL, self.dist = get_prediction_set_time(lastObservedPoint,elapsedTime,self.timeTransitionData,timeStep)
# else:


class trajectory_regression(path_regression):

    # Constructor
    def __init__(self, kernelX, kernelY, unit, stepUnit, finalArea, finalAreaAxis,linearPriorX=None, linearPriorY=None):
        self.speedRegressor = speed_regression(kernelX)
        pass
