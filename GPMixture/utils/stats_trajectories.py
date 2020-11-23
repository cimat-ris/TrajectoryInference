import numpy as np
import math
import statistics
import matplotlib.pyplot as plt
from utils.io_misc import euclidean_distance

""" Alternative functions, without the class trajectory """
#trajectory = [x,y,t]

#Trajectory arclen  ---  new path_arcLength from trajectory.py
def trajectory_arclength(tr):
    x, y = tr[0], tr[1]
    arclen = [0]
    for i in range(1,len(x)):
        d = euclidean_distance( [x[i],y[i]], [x[i-1], y[i-1]] )
        arclen.append(d)
        arclen[i] = arclen[i] + arclen[i-1]
        
    return arclen

#Average speed of a trajectory --- new path_speed from trajectory.py
def avg_speed(tr):
    x, y, t = tr[0], tr[1], tr[2]
    speed = []
    for i in range(1,len(x)):
        d = euclidean_distance( [x[i],y[i]], [x[i-1], y[i-1]] )
        dt = t[i] - t[i-1]
        speed.append(d/dt)
    
    return statistics.mean(speed)
    
def trajectory_duration(tr):
    t = tr[2]
    return t[-1] - t[0]


# Compute the average speed in a set of trajectories
#new get_pedestrian_average_speed
def pedestrian_avg_speed(trajectories):
    speed = []
    for tr in trajectories:
        speed.append(avg_speed(tr))
    
    return statistics.mean(speed)


#TODO: re-do histogram function for duration and arclength
def tr_histogram(trajectories):
    duration = []
    arclen = []
    
    for tr in trajectories:
        duration.append(trajectory_duration(tr))
        arclen.append(trajectory_arclength(tr))
        
"""*****************************************************"""

# Gets a set of trajectories and get the points (x,y,z)
def get_data_from_paths(trajectories):
    for i in range(len(trajectories)):
        auxX, auxY, auxT, auxS = trajectories[i].x, trajectories[i].y, trajectories[i].t, trajectories[i].s
        auxL = trajectories[i].l
        if(i==0):
            x, y, t, l, s = [auxX], [auxY], [auxT], [auxL], [auxS]
        else:
            x.append(auxX)
            y.append(auxY)
            t.append(auxT)
            l.append(auxL)
            s.append(auxS)
    return x, y, t, l, s

# Get a numpy vector of durations for trajectories, plus the min and max values
def get_min_and_max_duration(trajectories):
    n = len(trajectories)
    duration = np.zeros(n)
    maxDuration = 0
    minDuration = 10000

    for i in range(n):
        duration[i] = trajectories[i].duration
        if(duration[i] > maxDuration):
            # Determine max. duration
            maxDuration = duration[i]
        if(duration[i] < minDuration):
            # Determine min. duration
            minDuration = duration[i]
    return duration, minDuration, maxDuration

# Get a numpy vector of lengths of trajectories, plus the min and max values
def get_min_and_max_arclength(trajectories):
    n = len(trajectories)
    arcLen = []
    maxl = 0
    minl = 10000

    for i in range(n):
        arcLen.append(trajectories[i].length)
        if(arcLen[i] > maxl):
            maxl = arcLen[i]
        if(arcLen[i] < minl):
            minl = arcLen[i]
    return arcLen, minl, maxl


# Plot the histogram of duration and lengths
def histogram(trajectories):
    fig, (pht,phl) = plt.subplots(2, 1)
    vect = get_paths_duration(trajectories)
    vecl = get_paths_arclength(trajectories)
    _max = max(vect)
    # Taking bins of size 10
    numBins = int(_max/10)+1
    # Define the duration histogram
    ht = np.histogram(vect, bins = numBins)
    xt = []
    ytmin = []
    ytmax = []
    for i in range(len(ht[0])):
        xt.append(ht[1][i])
        ytmin.append(0)
        ytmax.append(ht[0][i])
    pht.vlines(xt,ytmin,ytmax,colors='b',linestyles='solid')
    pht.set_title('Distribution of durations')
    # Define the duration histogram
    hl    = np.histogram(vecl, bins = numBins)
    xl    = []
    ylmin = []
    ylmax = []
    for i in range(len(hl[0])):
        xl.append(hl[1][i])
        ylmin.append(0)
        ylmax.append(hl[0][i])
    phl.vlines(xl,ylmin,ylmax,colors='b',linestyles='solid')
    phl.set_title('Distribution of lengths')
    plt.show()

# Takes as an input a list of trajectories and outputs a vector with the corresponding times
def get_paths_duration(trajectories):
    __,__,t,__,__ = get_data_from_paths(trajectories)
    vec = []
    for i in range(len(trajectories)):
        N = len(t[i])
        vec.append(t[i][N-1] - t[i][0])
    return vec

# Takes as an input a list of trajectories and outputs a vector with the corresponding arc lengths
def get_paths_arclength(trajectories):
    # Get the x,y,arclengths
    __,__,__,l,__ = get_data_from_paths(trajectories)
    vec = []
    for i in range(len(trajectories)):
        N = len(l[i])
        vec.append(l[i][N-1])
    return vec

# Estimate the steps_units for a path set (unit = ratio between the distance to the goal and the length of the trajectory).
def get_steps_unit(pathSet):
    unit     = 0.0
    numPaths = len(pathSet)
    meanU = 0.0
    for path in pathSet:
        l = path.l[len(path.l)-1]
        if(l == 0):
            numPaths -= 1
        else:
            meanU += len(path.l)/l
    if(numPaths > 0):
        meanU = meanU/numPaths
    return meanU
