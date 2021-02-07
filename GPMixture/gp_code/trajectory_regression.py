"""
A class for GP-based trajectory regression (path AND time)
"""
import numpy as np
import math
from gp_code.path_regression import *


class trajectory_regression(path_regression):

    # Constructor
    def __init__(self, kernelX, kernelY, unit, stepUnit, finalArea, finalAreaAxis, modelSpeed, prior):
        # Init of the base class
        super(trajectory_regression, self).__init__(kernelX, kernelY, unit, stepUnit, finalArea, finalAreaAxis, prior)
