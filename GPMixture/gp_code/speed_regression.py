"""
A class for GP-based speed regression
"""
import numpy as np
import math
from gp_code.regression import *
from gp_code.path1D_regression import path1D_regression


class speed_regression(path1D_regression):
    def __init__(self, kernel):
        super().__init__(kernel, None)
