import numpy as np
import math
from gp_code.trajectory import trajectory
from utils.stats_trajectories import get_paths_arclength

"""********** FILTER PATHS **********"""
# Returns a matrix of trajectories:
# the entry (i,j) has the paths tha go from the goal i to the goal j
def define_trajectories_start_and_end_areas(startGoals, finishGoals, paths):
    # Number of starting goals
    nRows    = len(startGoals)
    # Number of ending goals
    nColumns = len(finishGoals)
    # Matrix to be built
    mat       = np.empty((nRows,nColumns),dtype=object)
    arclenMat = np.empty((nRows,nColumns),dtype=object)
    # Initialize the matrix elements to empty lists
    for i in range(nRows):
        for j in range(nColumns):
            mat[i][j]       = []
            arclenMat[i][j] = []
    # For all trajectories
    for path in paths:
        # Start and finish points
        startX, startY = path.x[ 0], path.y[ 0]
        endX, endY     = path.x[-1], path.y[-1]
        startIndex, endIndex = -1, -1
        # Determine which starting/ending indices they correspond to
        for j in range(nRows):
            if(is_in_area([startX,startY], startGoals[j])):
                startIndex = j
        for k in range(nColumns):
            if(is_in_area([endX,endY], finishGoals[k])):
                endIndex = k
        if(startIndex > -1 and endIndex > -1):
            # Keep the trajectory
            mat[startIndex][endIndex].append(path)
            # Keep the trajectory length
            arclenMat[startIndex][endIndex].append(path.length)
    return mat, arclenMat

# Computes the median m and the standard deviation s of a list of paths
# If any trajectory within this list differs with more than s from m, it is filtered out
def filter_paths(path_set):
    # Get the list of arclengths
    arclen_set = get_paths_arclength(path_set)
    # Median
    median_arclen = np.median(arclen_set)
    # Standard deviation
    SD = np.sqrt(np.var(arclen_set))
    # Resulting filtered set
    filtered_set = []
    for arclen,path in zip(arclen_set,path_set):
        if abs(arclen - median_arclen) <= 3.0*SD:
            filtered_set.append(path)
    return filtered_set

# Takes the start-goal matrix of lists of trajectories and filter them
# Output is:
# - matrix of filtered lists of trajectories
# - one big list of all the remaining trajectories
def filter_path_matrix(raw_path_set_matrix, nRows, mColumns):
    all_trajectories = []
    # Initialize a nRowsxnCols matrix with empty lists
    filtered_path_set_matrix = np.empty((nRows, mColumns),dtype=object)
    for i in range(nRows):
        for j in range(mColumns):
            filtered_path_set_matrix[i][j]=[]

    for i in range(nRows):
        for j in range(mColumns):
            # If the list of trajectories is non-empty, filter it
            if(len(raw_path_set_matrix[i][j]) > 0):
                filtered = filter_paths(raw_path_set_matrix[i][j])
                # Add the filtered trajectories
                # to the element list raw_path_set_matrix[i][j]
                for trajectory in filtered:
                    filtered_path_set_matrix[i][j].append(trajectory)
                    # Add the filtered trajectories to the list all_trajectories
                    all_trajectories.append(trajectory)
    return filtered_path_set_matrix, all_trajectories

# Filter paths that start and end in a goal zone
def get_paths_in_areas(paths, goals):
    useful = []
    for i in range(len(paths)):
        pathLen = len(paths[i].x)
        # Start pos
        first   = [paths[i].x[0],paths[i].y[0]]
        # End pos
        last    = [paths[i].x[pathLen-1],paths[i].y[pathLen-1]]
        isFirst, isLast = -1, -1
        # For all goal zones, check if start/end pos belongs to it
        for j in range(len(goals)):
            if(is_in_area(first,goals[j])):
                isFirst = j
            if(is_in_area(last,goals[j])):
                isLast = j
        # Filter
        if(isFirst > -1 and isLast > -1 and pathLen > 3):
            useful.append(paths[i])
    return useful

def get_path_set_given_time_interval(paths, startT, finishT):
    if(len(paths) == 0):
        print("empty set")
        return []
    pathSet = []
    i = 0
    t = startT
    while(t <= finishT):
        t = paths[i].t[0]
        if(startT <= t and t <= finishT):
            pathSet.append(paths[i])
        i+=1
    n = len(pathSet)
    for j in range(0):#n):
        print("[pathTime]:", paths[j].t)
    return pathSet

"""********** GOAL RELATED FUNCTIONS **********"""
def startGoal(p,goals):
    x, y = p.x[0], p.y[0]
    for i in range(len(goals)):
        if is_in_area(x,y,goals[i]):
            return i
    return -1

def get_goal_sequence(p, goals):
    g = []
    for i in range(len(p.x)):
        for j in range(len(goals)):
            xy = [p.x[i], p.y[i]]
            if is_in_area(xy, goals[j])==1:
                if len(g) == 0:
                    g.append(j)
                else:
                    if j != g[len(g)-1]:
                        g.append(j)
    return g

def getMultigoalPaths(paths,goals):
    N = len(paths)
    p = []
    g = []
    for i in range(N):
        gi = get_goal_sequence(paths[i],goals)
        if len(gi) > 2:
            p.append(i)
            g.append(gi)
    return p, g

def break_multigoal_path(multigoalPath, goalVec, goals):
    newPath = []
    p = multigoalPath
    g = goalVec
    newX, newY, newT = [], [], []
    goalInd = 0
    for j in range(len(p.x)):
        newX.append(p.x[j])
        newY.append(p.y[j])
        newT.append(p.t[j])

        if goalInd < len(g)-1:
            nextgoal = g[goalInd+1]
            xy = [p.x[j], p.y[j]]
            if is_in_area(xy,goals[nextgoal]):
                new = trajectory(newT,newX,newY)
                newPath.append(new)
                newX, newY, newT = [p.x[j]], [p.y[j]], [p.t[j]]
                goalInd += 1
    return newPath


def get_goal_center_and_boundaries(goal):
    points = []
    p, __ = middle_of_area(goal)
    points.append(p)
    lenX = goal[len(goal) -2] - goal[0]
    lenY = goal[len(goal) -1] - goal[1]
    q1 = [p[0]-lenX/2, p[1]]
    q2 = [p[0], p[1]+lenY/2]
    q3 = [p[0]+lenX/2, p[1]]
    q4 = [p[0], p[1]-lenY/2]
    points.append(q1)
    points.append(q2)
    points.append(q3)
    points.append(q4)
    return points


def get_goal_of_point(p, goals):
    for i in range(len(goals)):
        if(is_in_area(p,goals[i])):
            return i
    return -1

def get_path_start_goal(observedPath, goals):
    initPoint = [observedPath.x[0], observedPath.y[0]]
    for i in range(len(goals)):
        if(is_in_area(initPoint,goals[i])):
            return i

def get_path_finish_goal(observedPath, goals):
    n = len(observedPath.x)
    finishPoint = [observedPath.x[n-1], observedPath.y[n-1]]
    for i in range(len(goals)):
        if(is_in_area(finishPoint,goals[i])):
            return i
    return -1

# Centroid of an area
def middle_of_area(rectangle):
    n = len(rectangle)
    dx, dy = rectangle[n-2]-rectangle[0], rectangle[n-1]-rectangle[1]
    middle = [rectangle[0] + dx/2., rectangle[1] + dy/2.]
    return middle

# Centroid and size of an area
def goal_center_and_size(R):
    n = len(R)
    dx, dy = R[n-2]-R[0], R[n-1]-R[1]
    center = [R[0] + dx/2., R[1] + dy/2.]
    size = [dx, dy]
    return center, size

"""********** LINEAR PRIOR MEAN **********"""
# Linear regression: for data l,f(l), the function returns a, b for the line between the
# starting and ending points
def get_line_parameters(path, flag):
    n = len(path.l)
    ln = path.l[n-1]
    if(ln == 0):
        return 0., 0.
    if(flag == 'x'):
        b = path.x[0]
        a = (path.x[n-1]-b)/ln
    if(flag == 'y'):
        b = path.y[0]
        a = (path.y[n-1]-b)/ln
    return a, b

# Determine the covariance between line parameters, for a pair of goals
def get_line_covariance(lineParameters, mean):
    n = len(lineParameters)
    sum_ = 0.
    for i in range(n):
        sum_ += (lineParameters[i][0] - mean[0])*(lineParameters[i][1] - mean[1])
    if (n>1):
        cov = sum_/(n-1)
    else:
        cov=0.0
    return cov

# Determine the variances on line parameters, for a pair of goals
def get_line_variances(lineParameters, mean):
    n = len(lineParameters)
    sum_a, sum_b = 0., 0.
    for i in range(n):
        sum_a += (lineParameters[i][0] - mean[0])**2
        sum_b += (lineParameters[i][1] - mean[1])**2
    var = [sum_a/n, sum_b/n]
    # TODO: change this hard-coded value
    if var[0]<0.001:
        var[0]=0.001
    if var[1]<0.001:
        var[1]=0.001
    return var

# Takes as an input a set of trajectories (between goals) and a flag that says whether the orientation
# is in x or y
# Returns the mean value of the line parameters (to be used as a prior)
def get_linear_prior_mean(paths, flag):
    n = len(paths)
    if(n == 0):
        return [0.,0.,0.]

    lineParameters = []
    sum_a = 0.
    sum_b = 0.
    for i in range(n):
        # For each path, get the corresponding line parameters
        a, b = get_line_parameters(paths[i], flag)
        lineParameters.append([a,b])
        sum_a += a
        sum_b += b
    # Get the mean parameters
    mean = [sum_a/n, sum_b/n]
    # Get the covariance and standard deviations
    cov          = get_line_covariance(lineParameters, mean)
    vars         = get_line_variances(lineParameters, mean)
    return mean, cov, vars

"""********** HELPFUL FUNCTIONS **********"""
def equal(vx,vy,x,y):
    N = len(vx)
    if N == 0:
        return 0

    if vx[N-1] == x and vy[N-1] == y:
        return 1
    else:
        return 0

# Test if a point (x,y) belongs to an area R
def is_in_area(p,R):
    x = p[0]
    y = p[1]
    if(x >= R[0] and x <= R[len(R)-2]):
        if(y >= R[1] and y <= R[len(R)-1]):
            return 1
        else:
            return 0
    else:
        return 0

# Euclidean distance
def euclidean_distance(p, q): #p = (x,y)
    dist = math.sqrt( (p[0]-q[0])**2 + (p[1]-q[1])**2 )
    return dist

def copy_unitMat(unitMat, nGoals, nSubgoals):
    mat = []
    m = int(nSubgoals/nGoals)
    for i in range(nGoals):
        r = []
        for j in range(nSubgoals):
            k = int(j/m)
            r.append(unitMat[i][k])
        mat.append(r)
    return mat

def time_compare(path):
    return path.t[0]

def column(matrix, i):
    return [row[i] for row in matrix]

#predictedMeans es una lista que en i contiene array([X Y L]), esta funcion regresa una lista con [X, Y] en i
def get_prediction_arrays(predictedMeans):
    n = len(predictedMeans)
    XYvec = []
    for i in range(n):
        x = predictedMeans[i][:,0]
        y = predictedMeans[i][:,1]
        XYvec.append([x,y])
    return XYvec

"""********** ARC LENGHT TO TIME **********"""
def arclen_to_time(initTime,l,speed):
    t = [initTime]
    #print("acrlen to time l:",l)
    for i in range(1,len(l)):
        time_i = int(t[i-1] +(l[i]-l[i-1])/speed)
        t.append(time_i)
    return t

"""********** GET OBSERVED DATA **********"""
# Function to get the ground truth data: knownN data
def get_known_set(x,y,z,knownN):
    trueX,trueY, trueZ = [],[],[]
    knownN = int(knownN)
    for j in range(knownN): #numero de datos observados
        trueX.append(x[j])
        trueY.append(y[j])
        trueZ.append(z[j])

    return trueX, trueY, trueZ

def get_partial_path(fullPath, knownN):
    x,y,t = fullPath.x[0:knownN], fullPath.y[0:knownN], fullPath.t[0:knownN]
    partialPath = path(t,x,y)
    return partialPath

def get_observed_path_given_current_time(fullPath, currentTime):
    x, y, t= [],[],[]
    knownN = 0
    while(fullPath.t[knownN] <= currentTime):
        knownN += 1
    x,y,t = fullPath.x[0:knownN], fullPath.y[0:knownN], fullPath.t[0:knownN]
    observedPath = path(t,x,y)
    return observedPath
