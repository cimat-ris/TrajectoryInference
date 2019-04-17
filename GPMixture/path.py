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
        self.l = path_arcLength(self.x,self.y)
        self.duration, self.length = statistics(self.t,self.x,self.y)
        if self.duration > 0:            
            self.v = self.length/self.duration
        else:
            self.v = -1.

def statistics(t,x,y):
    duration = t[len(t)-1] - t[0]
        
    l = path_arcLength(x,y)
    length = l[len(l)-1]
    
    return duration, length

def path_arcLength(x,y):
    n = len(x)
    l = [0]
    for i in range(n):
        if i > 0:
            l.append(np.sqrt( (x[i]-x[i-1])**2 + (y[i]-y[i-1])**2 ) )
    for i in range(n):
        if(i>0):
            l[i] = l[i] +l[i-1]
    return l    
    
#************************************************************#
