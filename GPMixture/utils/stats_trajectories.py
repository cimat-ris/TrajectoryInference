import numpy as np
import matplotlib.pyplot as plt
import math

#This functions assume that trajectory = [x,y,t]

# Euclidean distance
def euclidean_distance(p, q): #p = (x,y)
    dist = math.sqrt( (p[0]-q[0])**2 + (p[1]-q[1])**2 )
    return dist

#Trajectory arclen  ---  new path_arcLength from trajectory.py
# TODO: Vectorization!
def trajectory_arclength(tr):
    x, y = tr[0], tr[1]
    arclen = [0]
    for i in range(1,len(x)):
        d = euclidean_distance( [x[i],y[i]], [x[i-1], y[i-1]] )
        arclen.append(d)
        arclen[i] = arclen[i] + arclen[i-1]

    return np.array(arclen)

# Average speed of a trajectory
def avg_speed(tr):
    x, y, t = np.array(tr[0]), np.array(tr[1]), np.array(tr[2])
    speed = np.divide(np.sqrt(np.square(x[1:]-x[:-1])+np.square(y[1:]-y[:-1])),t[1:]-t[:-1])
    return np.mean(speed)

# Median speed of a trajectory
def median_speed(tr):
    x, y, t = np.array(tr[0]), np.array(tr[1]), np.array(tr[2])
    speed = np.divide(np.sqrt(np.square(x[1:]-x[:-1])+np.square(y[1:]-y[:-1])),t[1:]-t[:-1])
    return np.median(speed)

# Trajectory duration
def trajectory_duration(tr):
    t = tr[2]
    return t[-1] - t[0]


# Compute the average speed in a set of trajectories
#new get_pedestrian_average_speed
def pedestrian_avg_speed(trajectories):
    speed = []
    for tr in trajectories:
        speed.append(avg_speed(tr))

    return np.mean(speed)


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

# new get_steps_unit
# Returns the average ratio u = number of steps(points) / arclen
def steps_unit(trajectories):
    unit = []
    for tr in trajectories:
        arclen = tr[2]
        nSteps = len(tr[2])
        unit.append(nSteps/arclen)

    return np.mean(unit)

# Truncates a float f to n decimals
def truncate(f, n):
    s = '{}'.format(f)
    if 'e' in s or 'E' in s:
        return '{0:.{1}f}'.format(f, n)
    i, p, d = s.partition('.')
    
    return '.'.join([i, (d+'0'*n)[:n]])
    