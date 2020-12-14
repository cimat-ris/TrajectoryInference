import numpy as np
import statistics as stats
from utils.stats_trajectories import trajectory_arclength
from utils.io_misc import is_in_area

""" Alternative functions, without the class trajectory """
#trajectory = [x,y,t]

#New define_trajectories_start_and_end_areas
# Returns a matrix of trajectories:
# the entry (i,j) has the paths that go from the goal i to the goal j
def separate_trajectories_between_goals(trajectories, goals):
    nGoals = len(goals)
    mat    = np.empty((nGoals,nGoals),dtype=object)
    # Initialize the matrix elements to empty lists
    for i in range(nGoals):
        for j in range(nGoals):
            mat[i][j]       = []
    # For all trajectories
    for tr in trajectories:
        x, y = tr[0], tr[1]
        trLen = len(x)
        if trLen > 2:
            # Start and finish points
            startX, startY = x[0], y[0]
            endX, endY     = x[-1], y[-1]
            startIndex, endIndex = -1, -1
            # Find starting and finishing goal
            for j in range(nGoals):
                if(is_in_area([startX,startY], goals[j])):
                    startIndex = j
            for k in range(nGoals):
                if(is_in_area([endX,endY], goals[k])):
                    endIndex = k
            if(startIndex > -1 and endIndex > -1):
                # Keep the trajectory
                mat[startIndex][endIndex].append(tr)

    return mat

#new filter_paths
# Computes the median and SD of a trajectory set
# Removes trajectories that differ more than 3SD
def filter_trajectories(trajectories):
    if len(trajectories) == 0:
        return []

    arclen = []
    for tr in trajectories:
        vec_arclen = trajectory_arclength(tr)
        tr_arclen  = vec_arclen[-1]
        arclen.append(tr_arclen)

    M  = stats.median(arclen)
    if len(arclen) < 2:
        SD = 0.0
    else:
        SD = stats.stdev(arclen)

    filtered_set = []
    for i in range(len(trajectories)):
        if arclen[i] > 0 and abs(arclen[i] - M) <= 3.0*SD:
            filtered_set.append(trajectories[i])

    return filtered_set

#new filter_path_matrix
# Output is:
# - matrix of filtered lists of trajectories
# - one big list of all the remaining trajectories
def filter_traj_matrix(raw_path_set_matrix, nRows, mColumns):
    all_trajectories = []
    # Initialize a nRowsxnCols matrix with empty lists
    filtered_matrix = np.empty((nRows, mColumns),dtype=object)
    for i in range(nRows):
        for j in range(mColumns):
            filtered_matrix[i][j] = []

    for i in range(nRows):
        for j in range(mColumns):
            # If the list of trajectories is non-empty, filter it
            if(len(raw_path_set_matrix[i][j]) > 0):
                filtered = filter_trajectories(raw_path_set_matrix[i][j])

                filtered_matrix[i][j].extend(filtered)
                all_trajectories.extend(filtered)

    return filtered_matrix, all_trajectories

#Gets a set of trj that start in a given time interval
#The list of trajectories is sorted by their initial time
def get_trajectories_given_time_interval(trajectories, startT, finishT):
    nTr = len(trajectories)

    if(nTr == 0):
        print("Empty set")
        return []

    trSet = []
    i = 0
    t = startT
    while(t <= finishT):
        tr = trajectories[i]
        t = tr[2][0]
        if(startT <= t and t <= finishT):
            trSet.append(tr)
        i+=1

    for j in range(0):#len(trSet)):
        print("[pathTime]:", trSet[j][2])

    return trSet

# Determines the list of goals that a trajectory goes through
def traj_goal_sequence(tr, goals):
    goalSeq = []
    x, y = tr[0], tr[1]

    for i in range(len(x)):
        for j in range(len(goals)):
            xy = [x[i], y[i]]
            if is_in_area(xy, goals[j]):
                if len(goalSeq) == 0:
                    goalSeq.append(j)
                else:
                    if j != goalSeq[-1]:
                        goalSeq.append(j)
    return goalSeq

# Select those trajectories that go through more than 2 goals
def multigoal_trajectories(trajectories, goals):
    multigoal_tr = []
    for tr in trajectories:
        # Determines the list of goals that a trajectory goes through
        goalSeq = traj_goal_sequence(tr, goals)
        if len(goalSeq) > 2:
            multigoal_tr.append(tr)
    return multigoal_tr

# Split a trajectory into sub-trajectories between pairs of goals
def break_multigoal_traj(tr, goals):
    x, y, t = tr[0], tr[1], tr[2]
    trSet = []

    X, Y, T = [], [], []    #new trajectory
    lastG = None            #last goal
    for i in range(len(x) ):
        X.append(x[i])
        Y.append(y[i])
        T.append(t[i])

        xy = [x[i], y[i]]       #current position
        for j in range(len(goals)):
            # If the position lies in the goal zone
            if is_in_area(xy, goals[j]):
                if lastG is None:
                    lastG = j
                # Split the trajectory
                elif lastG != j:
                    trSet.append([X,Y,T] )
                    X, Y, T = [], [], [] #start a new trajectory

    return trSet

# Returns {xi,yi,li} where
# xi = Vector of values x of the traj i
def get_data_from_set(trajectories):
    X, Y, L = [], [], []
    for tr in trajectories:
        X.append(tr[0])
        Y.append(tr[1])
        L.append(trajectory_arclength(tr) )
    return X, Y, L


'''------------ Linear Prior Mean --------------'''
# Linear regression: for data l,f(l), the function returns a, b for the line between the
# starting and ending points
def line_parameters(traj, flag):
    traj_arclen = trajectory_arclength(traj)
    arclen = traj_arclen[-1]
    if arclen == 0:
        return 0.,0.

    x, y = traj[0], traj[1]
    if(flag == 'x'):
        b = x[0]
        a = (x[-1]-b)/arclen
    if(flag == 'y'):
        b = y[0]
        a = (y[-1]-b)/arclen
    return a, b


"""-----------------------------------------------------"""

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

# Takes as an input a set of trajectories (between goals)
# and a flag that says whether the orientation is in x or y
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
        #a, b = get_line_parameters(paths[i], flag)
        a, b = line_parameters(paths[i], flag)
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

"""---------- Arclenght to time ----------"""
def arclen_to_time(initTime,l,speed):
    t = [initTime]
    for i in range(1,len(l)):
        time_i = int(t[i-1] +(l[i]-l[i-1])/speed)
        t.append(time_i)
    return np.array(t)

"""-------- Get Observed Data -------------"""
#new: get_known_set
# Function to get the ground truth data: knownN data
def observed_data(x,y,z, knownN):
    obsX, obsY, obsZ = x[0:knownN], y[0:knownN], z[0:knownN]
    return obsX, obsY, obsZ

def observed_data_given_time(x,y,t,time):
    i = 0
    while(t[i] <= time):
        i =+ 1
    obsX, obsY, obsT = x[0:i], y[0:i], t[0:i]
    return obsX, obsY, obsT
