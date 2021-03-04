"""
Test GP misture | Trautman
"""
import random
from test_common import *
from gp_code.kernels import *
from gp_code.mixture_gpT import mixtureGPT
from utils.stats_trajectories import truncate

# Read the areas file, dataset, and form the goalsLearnedStructure object
goalsDescriptions= '../parameters/CentralStation_GoalsDescriptions.csv'
trajFile         = '../datasets/GC/Annotation/'
imgGCS           = '../imgs/train_station.jpg'

traj_dataset, goalsData, trajMat, __ = read_and_filter('GCS',goalsDescriptions,trajFile,use_pickled_data=True)

#I'm skipping the training for now

goalsData.kernelsX = create_kernel_matrix('combinedTrautman', goalsData.nGoals, goalsData.nGoals)
goalsData.kernelsY = create_kernel_matrix('combinedTrautman', goalsData.nGoals, goalsData.nGoals)

"""**********          Testing          ***********"""
gi, gj, k = 0, 6, 5
path = trajMat[gi][gj][k]

mgps     = mixtureGPT(gi,goalsData)

# Divides the trajectory in part_num parts and consider
part_num = 5
pathSize = len(path[0])

# For different sub-parts of the trajectory
for i in range(1,part_num-1):
    p = plotter("../imgs/train_station.jpg")
    p.plot_scene_structure(goalsData)

    knownN = int((i+1)*(pathSize/part_num)) #numero de datos conocidos
    observations = observed_data(path,knownN)
    """Multigoal prediction test"""
    print('[INF] Updating likelihoods')
    likelihoods = mgps.update(observations)
    print('[INF] Performing prediction')
    predictedXYVec,varXYVec = mgps.predict_path()
    
    """
    print('[INF] Plotting')
    p.plot_multiple_predictions_and_goal_likelihood(pathX,pathY,knownN,goalsData.nGoals,likelihoods,predictedXYVec,varXYVec)
    print("[RES] Goals likelihood\n",mgps.goalsLikelihood)
    print("[RES] Mean likelihood:", mgps.meanLikelihood)
    print('[INF] Generating samples')
    vecX,vecY,__ = mgps.sample_paths(nSamples)
    trueX = observations[:,0]
    trueY = observations[:,1]
    p.plot_path_samples_with_observations(trueX.reshape((-1,1)),trueY.reshape((-1,1)),vecX,vecY)
    p.show() """

#print('--- goal transition ---')
#print(goalsData.priorTransitions)
"""
for i in range(goalsData.nGoals):
    s = np.sum(goalsData.priorTransitions[i])
    if s != 1.0:
        print('--- goal ',i,' ---')
        print('sum = ', s)
        #for val in goalsData.priorTransitions[i]: 
        #    print('original value:', val)#, '--- truncate:', truncate(val,16))
        
    """