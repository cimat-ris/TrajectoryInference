# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 15:13:31 2016

@author: karenlc
"""
import numpy as np
import math

class path:
    # Constructor
    def __init__(self,vecT,vecX,vecY):
        self.t = vecT.copy()
        self.x = vecX.copy()
        self.y = vecY.copy()
        self.l = path_arcLength(self.x,self.y)
        self.duration, self.length = statistics(self.t,self.x,self.y)
        if self.duration > 0:
            self.speed = self.length/self.duration
        else:
            self.speed = 0.

    def get_trajectory_from_path(self,x,y,l,speed):
        initTime     = self.t[0]
        arcLenToTime = arclen_to_time(initTime,l,speed)
        for i in range(len(x)):
            self.x.append( int(x[i]) )
            self.y.append( int(y[i]) )
            self.l.append( int(l[i]) )
            self.t.append(arcLenToTime[i])
        return self

    def join_path_with_sample(self,sampleX,sampleY,sampleL,speed):
        self.x = self.x + list(sampleX[:,0])
        self.y = self.y + list(sampleY[:,0])
        #print("NewY:",self.x)
        self.l = self.l + sampleL
        for i in range(len(sampleL)):
            self.t.append( int(sampleL[i]/speed) )
        #print("NewT:",self.t)


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
