"""
Created on Fri Mar 19 19:46:05 2021

@author: karen
"""
import random
from test_common import *
from gp_code.kernels import *
from gp_code.mixture_gpT import mixtureGPT
from utils.manip_trajectories import start_time, get_trajectories_given_time_interval, get_goal_of_point

# Read the areas file, dataset, and form the goalsLearnedStructure object
goalsDescriptions= '../parameters/CentralStation_GoalsDescriptions.csv'
trajFile         = '../datasets/GC/Annotation/'
imgGCS           = '../imgs/train_station.jpg'

traj_dataset, goalsData, trajMat, filtered = read_and_filter('GCS',goalsDescriptions,trajFile,use_pickled_data=True)

goalsData.kernelsX = create_kernel_matrix('combinedTrautman', goalsData.nGoals, goalsData.nGoals)
goalsData.kernelsY = create_kernel_matrix('combinedTrautman', goalsData.nGoals, goalsData.nGoals)

"""**********          Testing          ***********"""

# Sort by start time
filtered.sort(key=start_time)
# Find trajectories within an interval
trajs = get_trajectories_given_time_interval(filtered, 0, 100)
print('Number of trajs: ', len(trajs))

startGoals = []
for traj in trajs:
    p = [traj[0][0], traj[1][0] ]
    startGoals.append( get_goal_of_point(p, goalsData.areas_coordinates) )
    
print('Goals of trajectories:')
print( np.unique(startGoals) )
goalIndex = np.unique(startGoals)

mgps = [mixtureGPT(i,goalsData) for i in goalIndex ]
observations = np.empty(goalIndex.size )
nSamples = 5

part_num = 3
observedTime = 100

# For different sub-parts of the trajectory
for i in range(1,part_num-1):
    p = plotter("../imgs/train_station.jpg")
    p.plot_scene_structure(goalsData)

    knownN = int((i+1)*(observedTime/part_num)) #numero de datos conocidos
    
    #get observations of each trajectory
    #compute samples
    #Evaluate likelihood of samples
   
    #observations = observed_data(path,knownN)
    """Multigoal prediction test"""
    #print('[INF] Updating likelihoods')
    #likelihoods = mgps.update(observations)

    #print('[INF] Performing prediction')
    #predictedXYVec,varXYVec = mgps.predict_path()
    #print('[INF] Plotting')
    #p.plot_multiple_predictions_and_goal_likelihood(path[0],path[1],knownN,goalsData.nGoals,likelihoods,predictedXYVec,varXYVec)

    #print("[RES] Goals likelihood\n",mgps.goalsLikelihood)
    #print("[RES] Mean likelihood:", mgps.meanLikelihood)
    #print('[INF] Generating samples')
    #samplePaths = mgps.sample_paths(nSamples)
    #p.plot_path_samples_with_observations(observations,samplePaths)
    #p.show() 


