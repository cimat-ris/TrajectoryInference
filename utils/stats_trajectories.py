"""
For these functions, a trajectory has the shape [x,y,t]
"""
import numpy as np
import matplotlib.pyplot as plt
import math

# Evaluates the Euclidean distance
def euclidean_distance(p, q):
    dist = math.sqrt( (p[0]-q[0])**2 + (p[1]-q[1])**2 )
    return dist

# Evaluates vector of arclengths along a trajectory
def trajectory_arclength(trajectory):
    n        = len(trajectory)
    arclen   = np.empty(n)
    arclen[0]= 0
    for i in range(1,n):
        arclen[i] = arclen[i-1] + euclidean_distance(trajectory[i],trajectory[i-1])
    return arclen

# Evaluates vector of arclengths along a trajectory
def trajectory_speeds(trajectory):
    x, y, t = np.array(trajectory[:,0]), np.array(trajectory[:,1]), np.array(trajectory[:,2])
    return np.divide(np.sqrt(np.square(x[1:]-x[:-1])+np.square(y[1:]-y[:-1])),t[1:]-t[:-1])

# Evaluates the average speed along a trajectory
def avg_speed(trajectory):
    speed = np.divide(np.sqrt(np.square(trajectory[1:,0]-trajectory[:-1,0])+np.square(trajectory[1:,1]-trajectory[:-1,1])),trajectory[1:,2]-trajectory[:-1,2])
    return np.mean(speed)

# Evaluates the median speed along a trajectory
def median_speed(trajectory):
    speed = np.divide(np.sqrt(np.square(trajectory[1:,0]-trajectory[:-1,0])+np.square(trajectory[1:,1]-trajectory[:-1,1])),trajectory[1:,2]-trajectory[:-1,2])
    return np.median(speed)

# Evaluates the trajectory duration
def trajectory_duration(trajectory):
    return trajectory[-1,2] - trajectory[0,2]

# Evaluates histogram function for duration and arclength
def tr_histogram(trajectories):
    duration = []
    arclen   = []
    for trajectory in trajectories:
        duration.append(trajectory_duration(trajectory))
        arclen.append(trajectory_arclength(trajectory)[-1])

    fig, ax = plt.subplots(1,2)
    ax[0].hist(duration, bins='auto')
    ax[0].title.set_text("Histogram of trajectory duration")
    ax[1].hist(arclen, bins='auto')
    ax[1].title.set_text("Histogram of arclengths")
    plt.show()

# Truncates a float f to n decimals
def truncate(f, n):
    s = '{}'.format(f)
    if 'e' in s or 'E' in s:
        return '{0:.{1}f}'.format(f, n)
    i, p, d = s.partition('.')

    return '.'.join([i, (d+'0'*n)[:n]])
