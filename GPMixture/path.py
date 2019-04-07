# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 15:13:31 2016

@author: karenlc
"""
import numpy as np
import math
from dataManagement import *

class path:
    def __init__(self,vecT,vecX,vecY):
        self.t = vecT
        self.x = vecX
        self.y = vecY
        self.l = pathArcLength(self.x,self.y)
        self.duration, self.length = statistics(self.t,self.x,self.y)
        if self.duration > 0:            
            self.v = self.length/self.duration
        else:
            self.v = -1.

#************************************************************#
