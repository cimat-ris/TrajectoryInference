import numpy as np
import math

# Gets a set of paths and get the points (x,y,z)
# z = {time, arc-len} according to flag = {"time", "length"}
def get_data_from_paths(paths, flag):
    for i in range(len(paths)):
        auxX, auxY, auxT = paths[i].x, paths[i].y, paths[i].t
        auxL = arclength(auxX, auxY)
        if(i==0):
            x, y, t = [auxX], [auxY], [auxT]
            l = [auxL]
        else:
            x.append(auxX)
            y.append(auxY)
            t.append(auxT)
            l.append(auxL)

    if(flag == "time"):
        z = t
    if(flag == "length"):
        z = l
    return x, y, z

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

# Compute the average speed in a set of trajectories
def get_pedestrian_average_speed(trajectories):
    speed, validPaths = 0., 0
    for i in range(len(trajectories)):
        if trajectories[i].duration > 0:
            speed += trajectories[i].speed
            validPaths += 1
    avSpeed = speed/ validPaths
    return avSpeed

# Plot the histogram of duration or
def histogram(trajectories,flag):
    if flag == "duration":
        vec = get_paths_duration(trajectories)
    if flag == "length":
        vec = get_paths_arclength(trajectories)
    _max = max(vec)
    # Taking bins of size 10
    numBins = int(_max/10)+1
    # Define the histogram
    h = np.histogram(vec, bins = numBins)
    x = []
    ymin = []
    ymax = []
    for i in range(len(h[0])):
        x.append(h[1][i])
        ymin.append(0)
        ymax.append(h[0][i])
    plt.vlines(x,ymin,ymax,colors='b',linestyles='solid')

def get_paths_duration(trajectories):
    x,y,t = get_data_from_paths(trajectories,"time")
    vec = []
    for i in range(len(trajectories)):
        N = len(t[i])
        vec.append(t[i][N-1] - t[i][0])

    return vec

# Takes as an input a list of trajectories and outputs a vector with the corresponding total lengths
def get_paths_arclength(paths):
    # Get the x,y,arclengths
    x,y,z = get_data_from_paths(paths,"length")
    l = []
    for i in range(len(paths)):
        N = len(z[i])
        l.append(z[i][N-1])
    return l

# Computes the arclength of a curve described by a set of points (x,y)
def arclength(x,y):
    l = [0]
    for i in range(len(x)):
        if i > 0:
            l.append(np.sqrt( (x[i]-x[i-1])**2 + (y[i]-y[i-1])**2 ) )
    for i in range(len(x)):
        if(i>0):
            l[i] = l[i] +l[i-1]
    return l

def get_number_of_steps_unit(Mat, nGoals):
    unit = 0.0
    numUnits = 0
    for i in range(nGoals):
        for j in range(nGoals):
            numPaths = len(Mat[i][j])
            meanU = 0.0
            for k in range(numPaths):
                path = Mat[i][j][k]
                l = path.l[len(path.l)-1]
                if(l == 0):
                    numPaths -= 1
                else:
                    stps = len(path.l)
                    u = stps/l
                    meanU += u
            if(numPaths > 0):
                meanU = meanU/numPaths
            if(meanU >0):
                unit += meanU
                numUnits += 1
    unit = unit/numUnits
    return unit
