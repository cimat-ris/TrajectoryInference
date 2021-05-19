"""
Test GP misture | Trautman
"""
import random
from test_common import *
from gp_code.kernels import *
from gp_code.mixture_gpT import mixtureGPT
from utils.stats_trajectories import truncate

# Read the areas file, dataset, and form the goalsLearnedStructure object
trajFile         = 'datasets/GC/'
imgGCS           = 'imgs/train_station.jpg'

traj_dataset, goalsData, trajMat, __ = read_and_filter('GCS',trajFile,use_pickled_data=True)

#I'm skipping the training for now

goalsData.kernelsX = create_kernel_matrix('combinedTrautman', goalsData.goals_n, goalsData.goals_n)
goalsData.kernelsY = create_kernel_matrix('combinedTrautman', goalsData.goals_n, goalsData.goals_n)

"""**********          Testing          ***********"""
gi, gj, k = 0, 6, 5
path = trajMat[gi][gj][k]

mgps     = mixtureGPT(gi,goalsData)
nSamples = 5

# Divides the trajectory in part_num parts and consider
part_num = 3
pathSize = len(path[0])

# For different sub-parts of the trajectory
for i in range(1,part_num-1):
    p = plotter()
    p.set_background("imgs/train_station.jpg")
    p.plot_scene_structure(goalsData)

    knownN = int((i+1)*(pathSize/part_num)) #numero de datos conocidos
    observations = observed_data(path,knownN)
    """Multigoal prediction test"""
    print('[INF] Updating likelihoods')
    likelihoods = mgps.update(observations)

    print('[INF] Performing prediction')
    predictedXYVec,varXYVec = mgps.predict_path()
    #print('[INF] Plotting')
    #p.plot_multiple_predictions_and_goal_likelihood(path[0],path[1],knownN,goalsData.nGoals,likelihoods,predictedXYVec,varXYVec)

    print("[RES] Goals likelihood\n",mgps.goalsLikelihood)
    print("[RES] Mean likelihood:", mgps.meanLikelihood)
    print('[INF] Generating samples')
    samplePaths = mgps.sample_paths(nSamples)
    p.plot_path_samples_with_observations(observations,samplePaths)
    p.show()
