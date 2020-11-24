import numpy as np
import matplotlib.pyplot as plt
import statistics
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
    _ = plt.hist(duration, bins='auto') 
    plt.title("Histogram of trajectory duration")
    plt.show()
        
"""-----------------------------------------------------"""

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
