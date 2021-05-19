from utils.stats_trajectories import trajectory_arclength
import statistics as stats
import numpy as np

# Returns a matrix of trajectories:
# the entry (i,j) has the paths that go from the goal i to the goal j
def separate_trajectories_between_goals(trajectories,goals_areas):
    goals_n = len(goals_areas)
    goals   = goals_areas[:,1:]
    mat     = np.empty((goals_n,goals_n),dtype=object)
    # Initialize the matrix elements to empty lists
    for i in range(goals_n):
        for j in range(goals_n):
            mat[i][j]       = []
    # For all trajectories
    for idx,tr in enumerate(trajectories):
        x, y = tr[0], tr[1]
        traj_len = len(x)
        if traj_len > 2:
            # Start and finish points
            start_x, start_y     = x[0], y[0]
            end_x, end_y         = x[-1],y[-1]
            start_goal, end_goal = None, None
            # Find starting and finishing goal
            for j in range(goals_n):
                if(is_in_area([start_x,start_y], goals[j])):
                    start_goal = j
            for k in range(goals_n):
                if(is_in_area([end_x,end_y], goals[k])):
                    end_goal = k
            if start_goal is not None and end_goal is not None:
                mat[start_goal][end_goal].append(tr)
    return mat

# Removes atypical trajectories
def filter_trajectories(trajectories):
    if len(trajectories) == 0:
        return []

    arclen = []
    for tr in trajectories:
        vec_arclen = trajectory_arclength(tr)
        tr_arclen  = vec_arclen[-1]
        arclen.append(tr_arclen)

    # compute the median and SD of the trajectory set
    M  = stats.median(arclen)
    if len(arclen) < 2:
        SD = 0.0
    else:
        SD = stats.stdev(arclen)

    # remove trajectories that differ more than 3SD
    filtered_set = []
    for i in range(len(trajectories)):
        if arclen[i] > 0 and abs(arclen[i] - M) <= 3.0*SD:
            filtered_set.append(trajectories[i])

    return filtered_set

# Removes atypical trajectories from a multidimensional array
def filter_traj_matrix(raw_path_set_matrix):
    all_trajectories = []
    # Initialize a nRowsxnCols matrix with empty lists
    filtered_matrix = np.empty(raw_path_set_matrix.shape,dtype=object)
    for i in range(raw_path_set_matrix.shape[0]):
        for j in range(raw_path_set_matrix.shape[1]):
            filtered_matrix[i][j] = []

    for i in range(raw_path_set_matrix.shape[0]):
        for j in range(raw_path_set_matrix.shape[1]):
            # If the list of trajectories is non-empty, filter it
            if(len(raw_path_set_matrix[i][j]) > 0):
                filtered = filter_trajectories(raw_path_set_matrix[i][j])
                filtered_matrix[i][j].extend(filtered)
                all_trajectories.extend(filtered)

    return filtered_matrix, all_trajectories

def start_time(traj):
    return traj[2][0]

def get_trajectories_given_time_interval(trajectories, start_time, finish_time):
    # Note: the list of trajectories is sorted by initial time
    n = len(trajectories)
    if n == 0:
        print("[ERR] Empty set")
        return []

    traj_set = []
    i = 0
    t = start_time
    while(t <= finish_time):
        tr = trajectories[i]
        t = tr[2][0]
        if(start_time <= t and t <= finish_time):
            traj_set.append(tr)
        i += 1

    return traj_set

# Split a trajectory into sub-trajectories between pairs of goals
def break_multigoal_traj(tr, goals):
    x, y, t = tr[0], tr[1], tr[2]
    traj_set = []
    new_x, new_y, new_t = [], [], []    # New trajectory
    last_goal = -1                      # Last goal
    started   = False                   # Flag to indicate that we have started with one goal
    for i in range(len(x)):
        xy = [x[i], y[i]]       # Current position
        # Am I in a goal
        current_goal = -1
        for j in range(len(goals)):
            # If the position lies in the goal zone
            if is_in_area(xy, goals[j,1:]):
                current_goal=j
        if current_goal==-1 and last_goal!=-1 and started:
            # Split the trajectory just before
            traj_set.append([np.array(new_x),np.array(new_y),np.array(new_t)] )
        if current_goal==-1 and last_goal!=-1:
            # At that point we start the trajectory
            # with a point that should be in last_goal
            started = True
            new_x, new_y, new_t = [x[i-1]], [y[i-1]], [t[i-1]]
        last_goal=current_goal
        new_x.append(x[i])
        new_y.append(y[i])
        new_t.append(t[i])
        # Coming at the end
        if current_goal>0 and i==len(x)-1 and started:
            traj_set.append([np.array(new_x),np.array(new_y),np.array(new_t)] )
    return traj_set

# Returns 3 lists with the x, y and arc-len values of a trajectory set, respectively
def get_data_from_set(trajectories):
    list_x, list_y, list_arclen = [], [], []
    for tr in trajectories:
        list_x.append(tr[0])
        list_y.append(tr[1])
        list_arclen.append(trajectory_arclength(tr) )
    return list_x, list_y, list_arclen


# Linear regression: f(l) = a + b*l
# Returns the slope of the line and the intercept
def line_parameters(traj, flag):
    traj_arclen = trajectory_arclength(traj)
    arclen = traj_arclen[-1]
    if arclen == 0:
        return 0.,0.

    x, y = traj[0], traj[1]
    if flag == 'x':
        b = x[0]
        a = (x[-1]-b)/arclen
    if flag == 'y':
        b = y[0]
        a = (y[-1]-b)/arclen
    return a, b

# Takes as an input a set of trajectories (between goals)
# and a flag that says whether the orientation is in x or y
def get_linear_prior_mean(trajectories, flag):
    n = len(trajectories)
    if n == 0:
        return [0.,0.,0.]
    lineParameters = np.array([ line_parameters(trajectories[i], flag) for i in range(n)])
    mean = [np.mean(lineParameters[:,0]), np.mean(lineParameters[:,1]) ]
    var = [np.var(lineParameters[:,0]), np.var(lineParameters[:,1]) ]
    cov = np.cov(lineParameters[:,0],lineParameters[:,1])

    return mean, var

def arclen_to_time(init_time, arclen, speed):
    n = len(arclen)
    time = np.zeros(n, dtype=int)
    time[0] = init_time
    for i in range(1,len(arclen)):
        time[i] = int(time[i-1] + (arclen[i]-arclen[i-1])/speed)

    return time


# Function to get the ground truth data: n data
def observed_data(traj, n):
    if (len(traj)==4):
        x, y, l, t = traj
        obsX, obsY, obsL, obsT = np.reshape(x[0:n],(-1,1)), np.reshape(y[0:n],(-1,1)), np.reshape(l[0:n],(-1,1)),np.reshape(t[0:n],(-1,1))
        obsS = np.reshape(np.divide(np.sqrt(np.square(x[1:n+1]-x[:n])+np.square(y[1:n+1]-y[:n])),t[1:n+1]-t[:n]),(-1,1))
        gtX, gtY, gtT = np.reshape(x[n:],(-1,1)), np.reshape(y[n:],(-1,1)),np.reshape(t[n:],(-1,1))
        gtS =  np.reshape(np.concatenate([np.divide(np.sqrt(np.square(x[n+1:]-x[n:-1])+np.square(y[n+1:]-y[n:-1])),t[n+1:]-t[n:-1]),[0.0]]),(-1,1))
        if gtS.shape[0]<2:
            return None, None
        gtS[-1,0] = gtS[-2,0]
        return np.concatenate([obsX, obsY, obsL, obsT, obsS],axis=1),np.concatenate([gtX, gtY, gtT,gtS],axis=1)
    else:
        if (len(traj)==3):
            x, y, t = traj
            obsX, obsY, obsT = np.reshape(x[0:n],(-1,1)), np.reshape(y[0:n],(-1,1)), np.reshape(t[0:n],(-1,1))
        return np.concatenate([obsX, obsY, obsT],axis=1)

def observed_data_given_time(traj, time):
    _, _, t = traj
    i = 0
    while(t[i] <= time and i < len(t)-1):
        i += 1
    return observed_data(traj, i)

def reshape_trajectory(traj):
    x, y, t = traj[:,0], traj[:,1], traj[:,2]
    x.reshape((-1,1))
    y.reshape((-1,1))
    t.reshape((-1,1))
    return [x,y,t]


# Checks if a point (x,y) belongs to an area R
def is_in_area(p, area):
    x = p[0]
    y = p[1]
    if(x >= min(area[0::2]) and x <= max(area[0::2])):
        if(y >= min(area[1::2]) and y <= max(area[1::2])):
            return True
        else:
            return False
    else:
        return False

def get_goal_of_point(p, goals):
    for i in range(len(goals)):
        if is_in_area(p,goals[i]):
            return i
    return None

# Returns the center of a rectangular area
def goal_centroid(area):
    dx, dy = area[-2] - area[0], area[-1] - area[1]
    centroid = [area[0] + dx/2., area[1] + dy/2.]
    return centroid

def goal_center_and_size(area):
    dx, dy = area[-2] - area[0], area[-1] - area[1]
    center = [area[0] + dx/2., area[1] + dy/2.]
    size = [dx, dy]
    return center, size

#TODO: check if these functions are useful
"""
def get_goal_center_and_boundaries(goal):
    p, __ = goal_centroid(goal)
    lenX = goal[len(goal) -2] - goal[0]
    lenY = goal[len(goal) -1] - goal[1]
    q1 = [p[0]-lenX/2, p[1]]
    q2 = [p[0], p[1]+lenY/2]
    q3 = [p[0]+lenX/2, p[1]]
    q4 = [p[0], p[1]-lenY/2]
    return [p,q1,q2,q3,q4]


#predictedMeans es una lista que en i contiene array([X Y L]), esta funcion regresa una lista con [X, Y] en i
def get_prediction_arrays(predictedMeans):
    n = len(predictedMeans)
    XYvec = []
    for i in range(n):
        x = predictedMeans[i][:,0]
        y = predictedMeans[i][:,1]
        XYvec.append([x,y])
    return XYvec
"""
