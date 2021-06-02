"""
For these functions, a trajectory has the shape [x,y,t]
"""
import numpy as np
import matplotlib.pyplot as plt
import math

# Euclidean distance
def euclidean_distance(p, q):
    dist = math.sqrt( (p[0]-q[0])**2 + (p[1]-q[1])**2 )
    return dist

def trajectory_arclength(traj):
    x, y = traj[0], traj[1]
    n    = len(x)
    arclen = np.empty(n)
    arclen[0] = 0
    for i in range(1,n):
        d = euclidean_distance( [x[i],y[i]], [x[i-1], y[i-1]] )
        arclen[i] = arclen[i-1] + d

    return arclen

def trajectory_speeds(traj):
    x, y, t = np.array(traj[0]), np.array(traj[1]), np.array(traj[2])
    return np.divide(np.sqrt(np.square(x[1:]-x[:-1])+np.square(y[1:]-y[:-1])),t[1:]-t[:-1])

# Average speed of a trajectory
def avg_speed(traj):
    x, y, t = np.array(traj[0]), np.array(traj[1]), np.array(traj[2])
    speed = np.divide(np.sqrt(np.square(x[1:]-x[:-1])+np.square(y[1:]-y[:-1])),t[1:]-t[:-1])
    return np.mean(speed)

# Median speed of a trajectory
def median_speed(traj):
    x, y, t = np.array(traj[0]), np.array(traj[1]), np.array(traj[2])
    speed = np.divide(np.sqrt(np.square(x[1:]-x[:-1])+np.square(y[1:]-y[:-1])),t[1:]-t[:-1])
    return np.median(speed)

# Trajectory duration
def trajectory_duration(traj):
    t = traj[2]
    return t[-1] - t[0]

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

# Truncates a float f to n decimals
def truncate(f, n):
    s = '{}'.format(f)
    if 'e' in s or 'E' in s:
        return '{0:.{1}f}'.format(f, n)
    i, p, d = s.partition('.')

    return '.'.join([i, (d+'0'*n)[:n]])
