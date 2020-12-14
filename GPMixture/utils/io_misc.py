import numpy as np
import math

def write_data(data, fileName):
    n = len(data)
    f = open(fileName,"w+")
    for i in range(n):
        s = "%d\n"%(data[i])
        f.write(s)
    f.close()

def read_data(fileName):
    data = []
    f = open(fileName,'r')
    lines = f.readlines()
    for i in range(len(lines)):
        lines[i] = lines[i].strip("\n")
        data.append( float(lines[i]) )
    f.close()
    return data

# Euclidean distance
def euclidean_distance(p, q): #p = (x,y)
    dist = math.sqrt( (p[0]-q[0])**2 + (p[1]-q[1])**2 )
    return dist

"""---------- Goal related functions ----------"""

# Checks if a point (x,y) belongs to an area R
def is_in_area(p,R):
    x = p[0]
    y = p[1]
    if(x >= R[0] and x <= R[-2]):
        if(y >= R[1] and y <= R[-1]):
            return 1
        else:
            return 0
    else:
        return 0
    
def get_goal_center_and_boundaries(goal):
    p, __ = goal_centroid(goal)
    lenX = goal[len(goal) -2] - goal[0]
    lenY = goal[len(goal) -1] - goal[1]
    q1 = [p[0]-lenX/2, p[1]]
    q2 = [p[0], p[1]+lenY/2]
    q3 = [p[0]+lenX/2, p[1]]
    q4 = [p[0], p[1]-lenY/2]
    return [p,q1,q2,q3,q4]


def get_goal_of_point(p, goals):
    for i in range(len(goals)):
        if(is_in_area(p,goals[i])):
            return i
    return -1

# Middle of a goal area
def goal_centroid(R):
    n = len(R)
    dx, dy = R[n-2]-R[0], R[n-1]-R[1]
    center = [R[0] + dx/2., R[1] + dy/2.]
    return center

# Centroid and size of an area
def goal_center_and_size(R):
    n = len(R)
    dx, dy = R[n-2]-R[0], R[n-1]-R[1]
    center = [R[0] + dx/2., R[1] + dy/2.]
    size = [dx, dy]
    return center, size