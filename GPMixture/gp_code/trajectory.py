# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 15:13:31 2016
@author: karenlc
"""
import numpy as np
import math

class trajectory:
    # Constructor
    def __init__(self,vecT,vecX,vecY):
        self.t = vecT.copy()
        self.x = vecX.copy()
        self.y = vecY.copy()
        self.l = self.path_arcLength()
        self.s = self.path_speed()
        self.duration, self.length = self.statistics()
        if self.duration > 0:
            self.averageSpeed = self.length/self.duration
        else:
            self.averageSpeed = 0.

    # Build a trajectory from a path
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

    # Compute some statistics on the trajectory
    def statistics(self):
        duration = self.t[len(self.t)-1] - self.t[0]
        l = self.path_arcLength()
        length = l[len(l)-1]
        return duration, length

    # Compute all arc-lengths
    def path_arcLength(self):
        n = len(self.x)
        l = [0]
        for i in range(n):
            if i > 0:
                l.append(np.sqrt( (self.x[i]-self.x[i-1])**2 + (self.y[i]-self.y[i-1])**2 ) )
        for i in range(n):
            if(i>0):
                l[i] = l[i] +l[i-1]
        return l

    # Compute all instantaneous speeds
    def path_speed(self):
        n      = len(self.x)
        speeds = [0,0]
        for i in range(n):
            if i > 1:
                dl    = np.sqrt( (self.x[i]-self.x[i-2])**2 + (self.y[i]-self.y[i-2])**2 )
                dt    = self.t[i]-self.t[i-2]
                speeds.append(dl/dt)
        if n>2:
            speeds[0]=speeds[2]
            speeds[1]=speeds[2]
        return speeds
