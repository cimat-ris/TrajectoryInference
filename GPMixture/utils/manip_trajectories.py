from utils.stats_trajectories import trajectory_arclength
import statistics as stats
import numpy as np
import math

""" Alternative functions, without the class trajectory """
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
            startG, endG = None, None
            # Find starting and finishing goal
            for j in range(nGoals):
                if(is_in_area([startX,startY], goals[j])):
                    startG = j
            for k in range(nGoals):
                if(is_in_area([endX,endY], goals[k])):
                    endG = k
            if(not startG == None and not endG == None):
                mat[startG][endG].append(tr)
    return mat


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

def start_time(traj):
    return traj[2][0]

#TODO: Gets a set of trj that start in a given time interval
#The list of trajectories is sorted by their initial time
def get_trajectories_given_time_interval(trajectories, startT, finishT):
    nTr = len(trajectories)

    if(nTr == 0):
        print("[ERR] Empty set")
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

# Takes as an input a set of trajectories (between goals)
# and a flag that says whether the orientation is in x or y
def get_linear_prior_mean(paths, flag):
    n = len(paths)
    if(n == 0):
        return [0.,0.,0.]
    lineParameters = np.array([ line_parameters(paths[i], flag) for i in range(n)])
    mean = [np.mean(lineParameters[:,0]), np.mean(lineParameters[:,1]) ]
    var = [np.var(lineParameters[:,0]), np.var(lineParameters[:,1]) ]
    cov = np.cov(lineParameters[:,0],lineParameters[:,1])

    return mean, var
"""-----------------------------------------------------"""

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
# Function to get the ground truth data: knownN data
def observed_data(traj, n):
    # TODO:
    if (len(traj)==4):
        x, y, l, t = traj
        obsX, obsY, obsL, obsT = np.reshape(x[0:n],(-1,1)), np.reshape(y[0:n],(-1,1)), np.reshape(l[0:n],(-1,1)),np.reshape(t[0:n],(-1,1))
        obsS = np.divide(np.sqrt(np.square(obsX[1:]-obsX[:-1])+np.square(obsY[1:]-obsY[:-1])),obsT[1:]-obsT[:-1])
        gtX, gtY, gtT = np.reshape(x[n:],(-1,1)), np.reshape(y[n:],(-1,1)),np.reshape(t[n:],(-1,1))
        return np.concatenate([obsX, obsY, obsL, obsT],axis=1),np.concatenate([gtX, gtY, gtT],axis=1)
    else:
        if (len(traj)==3):
            x, y, t = traj
            obsX, obsY, obsT = np.reshape(x[0:n],(-1,1)), np.reshape(y[0:n],(-1,1)), np.reshape(t[0:n],(-1,1))
        return np.concatenate([obsX, obsY, obsT],axis=1)


def observed_data_given_time(traj, time):
    _, _, t = traj
    i = 0
    while(t[i] <= time and i < len(t)-1 ):
        i += 1
    return observed_data(traj, i)

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
    return None

# Middle of a goal area
def goal_centroid(R):
    dx, dy = R[-2]-R[0], R[-1]-R[1]
    center = [R[0] + dx/2., R[1] + dy/2.]
    return center

# Centroid and size of an area
def goal_center_and_size(R):
    dx, dy = R[-2]-R[0], R[-1]-R[1]
    center = [R[0] + dx/2., R[1] + dy/2.]
    size = [dx, dy]
    return center, size

# TODO: this
def get_subgoals(n, goal, axis):
    c, size = goal_center_and_size(goal)
    d = 0
    if axis == 'x':
        d = size[0]
    else:
        d = size[1]
    sgSize = d/n
    #sg = [ for i in range(n)]

# TODO: REDO IN A BETTER WAY
def get_subgoals_areas(nSubgoals, goal, axis):
    goalDX = goal[len(goal) -2] - goal[0]
    goalDY = goal[len(goal) -1] - goal[1]
    goalCenterX = goal[0]+ goalDX/2.0
    goalCenterY = goal[1]+ goalDY/2.0
    goalMinX    = goal[0]
    goalMinY    = goal[1]
    goalMaxX    = goal[-2]
    goalMaxY    = goal[-1]
    subGoalsAreas = []
    if axis == 0:
        subgoalDX = goalDX/nSubgoals
        subgoalDY = goalDY
        for i in range(nSubgoals):
            subGoalsAreas.append( [goalMinX+i*subgoalDX,goalMinY,goalMinX+(i+1)*subgoalDX,goalMinY,goalMinX+i*subgoalDX,goalMaxY,goalMinX+(i+1)*subgoalDX,goalMaxY] )
    else:
        subgoalDX = goalDX
        subgoalDY = goalDY/nSubgoals
        _x = goalCenterX
        _y = goal[1]
        for i in range(nSubgoals):
            subGoalsAreas.append([goalMinX,goalMinY+i*subgoalDY,goalMaxX,goalMinY+i*subgoalDY,goalMinX,goalMinY+(i+1)*subgoalDY,goalMaxX,goalMinY+(i+1)*subgoalDY])

    return subGoalsAreas


def get_subgoals_center_and_size(nSubgoals, goal, axis):
    goalX = goal[len(goal) -2] - goal[0]
    goalY = goal[len(goal) -1] - goal[1]
    goalCenterX = goal[0]+ goalX/2
    goalCenterY = goal[1]+ goalY/2

    subgoalsCenter = []
    subgoalX, subgoalY = 0,0
    if axis == 0:
        subgoalX = goalX/nSubgoals
        subgoalY = goalY
        _x = goal[0]
        _y = goalCenterY
        for i in range(nSubgoals):
            subgoalsCenter.append( [_x+subgoalX/2.0, _y] )
            _x += subgoalX
    else:
        subgoalX = goalX
        subgoalY = goalY/nSubgoals
        _x = goalCenterX
        _y = goal[1]
        for i in range(nSubgoals):
            subgoalsCenter.append( [_x, _y+subgoalY/2.0] )
            _y += subgoalY

    return subgoalsCenter, [subgoalX, subgoalY]
