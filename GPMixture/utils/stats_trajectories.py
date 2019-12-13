import numpy as np
import math
import matplotlib.pyplot as plt

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

# Compute the average speed in a set of trajectories
def get_pedestrian_average_speed(trajectories):
    speed, validPaths = 0., 0
    for i in range(len(trajectories)):
        if trajectories[i].duration > 0:
            speed += trajectories[i].speed
            validPaths += 1
    avSpeed = speed/ validPaths
    return avSpeed

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

# TODO: should be removed and we should use an estimate of the velocity instead
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
