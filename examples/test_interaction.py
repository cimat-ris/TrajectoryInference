"""
Interaction potential test | Trautman
"""
## TODO: repair
import random
import numpy as np
from test_common import *
from gp_code.kernels import *
from gp_code.mGPt_trajectory_prediction import mGPt_trajectory_prediction
from gp_code.interactions import interaction_potential_for_a_set_of_trajectories
from utils.manip_trajectories import start_time, get_trajectories_given_time_interval, get_goal_of_point
from utils.manip_trajectories import observed_data_given_time, reshape_trajectory


# Read the areas file, dataset, and form the goalsLearnedStructure object
imgGCS           = './datasets/GC/reference.jpg'
coordinates      = "img"

traj_dataset, goalsData, trajMat, filtered = read_and_filter('GCS',coordinate_system=coordinates,use_pickled_data=True)

goalsData.kernelsX = create_kernel_matrix('combinedTrautman', goalsData.goals_n, goalsData.goals_n)
goalsData.kernelsY = create_kernel_matrix('combinedTrautman', goalsData.goals_n, goalsData.goals_n)

"""**********          Testing          ***********"""

# Sort by start time
print(filtered)
filtered.sort(key=start_time)
# Find trajectories within an interval
trajs = get_trajectories_given_time_interval(filtered, 0, 100)
n =  10#len(trajs)
print('Number of trajs: ', len(trajs))

startGoals = []
for traj in trajs:
    p = [traj[0][0], traj[1][0] ]
    startGoals.append( get_goal_of_point(p, goalsData.goals_areas[:,1:]) )

print('Goals of trajectories:')
print( np.unique(startGoals) )

mgps         = np.empty(n,dtype=object)
observations = np.empty(n,dtype=object)
samples      = np.empty(n,dtype=object)

for i in range(n):
    mgps[i] = mGPt_trajectory_prediction(startGoals[i] ,goalsData)

samples_n = 5
part_num = 3
observedTime = 100

# For different sub-parts of the trajectory
for i in range(1,part_num-1):
    p = plotter()
    if coordinates=='img':
        p.set_background(imgGCS)
    p.plot_scene_structure(goalsData)
    time = int((i+1)*(observedTime/part_num)) #numero de datos conocidos

    #get observations of each trajectory
    for j in range(n):
        print('------------ Trajectory ',j,'---------------')
        observations[j] = observed_data_given_time(trajs[j],time)
        print('[INF] Updating likelihoods')
        likelihoods = mgps[j].update(observations[j])
        print('[INF] Performing prediction')
        predictedXYVec,varXYVec = mgps[j].predict_path()
        print('[INF] Generating samples')
        samples[j] = mgps[j].sample_paths(samples_n)
        p.plot_path_samples_with_observations(observations[j],samples[j])
        p.show()

    #Evaluate likelihood of samples
    for j in range(samples_n):
        sample_set = []
        for k in range(n):
            sample_set.append(samples[k][j])
        print('[INF] Computing interaction potential')
        val = interaction_potential_for_a_set_of_trajectories(sample_set)#interaction_potential_using_approximation()
        print('interaction potential =',val)
