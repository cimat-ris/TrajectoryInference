from utils.stats_trajectories import trajectory_arclength
import statistics as stats
import numpy as np
import logging


def image_to_world(trajectory,homography):
    pp = np.stack((trajectory[:,0],trajectory[:,1],np.ones(len(trajectory))),axis=1)
    PP = np.matmul(homography, pp.T).T
    P_normal = PP / np.repeat(PP[:, 2].reshape((-1, 1)), 3, axis=1)
    return P_normal[:, :2]


# Returns a matrix of trajectories:
# the entry (i,j) has the paths that go from the goal i to the goal j
def separate_trajectories_between_goals(trajectories, goals_areas):
    goals_n = len(goals_areas)
    goals   = goals_areas[:,1:]
    mat     = np.empty((goals_n,goals_n),dtype=object)
    # Initialize the matrix elements to empty lists
    for i in range(goals_n):
        for j in range(goals_n):
            mat[i][j]       = []
    # For all trajectories
    for trajectory in trajectories:
        traj_len = len(trajectory)
        associated_to_goals = False
        if traj_len > 2:
            # Start and finish points
            start     = trajectory[0,:]
            end       = trajectory[-1,:]
            start_goal, end_goal = None, None
            # Find starting and finishing goal
            for j in range(goals_n):
                if(is_in_area(start, goals[j])):
                    start_goal = j
            for k in range(goals_n):
                if(is_in_area(end, goals[k])):
                    end_goal = k
            if start_goal is not None and end_goal is not None:
                mat[start_goal][end_goal].append(trajectory)
                associated_to_goals = True
    return mat

# Removes atypical trajectories
def filter_trajectories(trajectories):
    n_trajs = len(trajectories)
    if n_trajs == 0:
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

    # Remove atypical trajectories that differ more than 3SD
    filtered_set = []
    for i in range(n_trajs):
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
    return traj[0,2]

def end_time(traj):
    return traj[-1,2]

# For a set of trajectories, determine those that have some timestamps in the [start_time, finish_time] interval
def get_trajectories_given_time_interval(trajectories, start_time, finish_time):
    # Note: the list of trajectories should be initially sorted by initial time
    n = len(trajectories)
    if n == 0:
        logging.error("Empty set")
        return []

    traj_set = []
    i = 0
    t = start_time
    while (t <= finish_time and i<n):
        tr = trajectories[i]
        # Starting/ending times
        st  = tr[0,2]
        et  = tr[-1,2]
        if(start_time <= et and st <= finish_time):
            traj_set.append(tr)
        i += 1
    return traj_set

# Split a trajectory into sub-trajectories between pairs of goals
def break_multigoal_traj(tr, goals):
    traj_set = []
    new_traj = []          # New trajectory
    last_goal = -1         # Last goal
    started   = False      # Flag to indicate that we have started with one goal
    for i,pos in enumerate(tr):
        # Am I within a goal area
        current_goal = -1
        for j in range(len(goals)):
            # If the position lies in the goal zone
            if is_in_area(pos[0:2], goals[j,1:]):
                current_goal=j
        # We have just left a goal zone
        if current_goal==-1 and last_goal!=-1:
            if started:
                # Split the trajectory just before and add the piece
                new_traj = np.array(new_traj)
                traj_set.append(new_traj)
            # At that point we start the trajectory
            # with a point that should be in last_goal
            started = True
            new_traj = [tr[i-1]]
        last_goal=current_goal
        new_traj.append(pos)
        # Coming at the end
        if current_goal>0 and i==len(tr)-1 and started:
            new_traj = np.array(new_traj)
            traj_set.append(new_traj)
    return traj_set

# Returns 3 lists with the x, y and arc-len values of a trajectory set, respectively
def get_data_from_set(trajectories):
    list_x, list_y, list_arclen = [], [], []
    for trajectory in trajectories:
        list_x.append(trajectory[:,0])
        list_y.append(trajectory[:,1])
        list_arclen.append(trajectory_arclength(trajectory))
    return list_x, list_y, list_arclen

# Linear regression: f(l) = a + b*l
# Returns the slope of the line and the intercept
def line_parameters(trajectory, flag):
    arclen = trajectory_arclength(trajectory)[-1]
    if arclen == 0:
        return 0.,0.
    x, y = trajectory[:,0], trajectory[:,1]
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
    lineParameters = np.array([line_parameters(trajectories[i], flag) for i in range(n)])
    mean = [np.median(lineParameters[:,0]), np.median(lineParameters[:,1]) ]
    var  = [np.var(lineParameters[:,0]), np.var(lineParameters[:,1]) ]
    cov  = np.cov(lineParameters[:,0],lineParameters[:,1])
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
    x, y = p[0], p[1]

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
def goal_center(area):
    dx, dy = area[-2] - area[0], area[-1] - area[1]
    centroid = [area[0] + dx/2., area[1] + dy/2.]
    return centroid

def goal_center_and_size(area):
    center = np.array([0.25*float(np.sum(area[::2])),0.25*float(np.sum(area[1::2]))])
    size = np.array([float(np.max(area[::2]))-float(np.min(area[::2])),float(np.max(area[1::2]))-float(np.min(area[1::2]))])
    return center, size
