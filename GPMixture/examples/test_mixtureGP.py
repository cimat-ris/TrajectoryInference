"""
@author: karenlc
"""
from test_common import *
from gp_code.mixture_gp import mixtureOfGPs


# Read the areas file, dataset, and form the goalsLearnedStructure object
goalsDescriptions= 'parameters/CentralStation_GoalsDescriptions.csv'
trajFile         = 'datasets/GC/Annotation/'
imgGCS           = 'imgs/train_station.jpg'

traj_dataset, goalsData, trajMat, __ = read_and_filter('GCS',goalsDescriptions,trajFile,use_pickled_data=True)

# Selection of the kernel type
kernelType = "linePriorCombined"
nParameters = 4

# Read the kernel parameters from file
goalsData.kernelsX = read_and_set_parameters('parameters/linearpriorcombined20x20_x.txt',nParameters)
goalsData.kernelsY = read_and_set_parameters('parameters/linearpriorcombined20x20_y.txt',nParameters)

"""******************************************************************************"""
"""**************    Testing                           **************************"""
# We give the start and ending goals and the index of the trajectory to predict
startG = 0
nextG  = 6
pathId = 3

# Get the ground truth path
_path = trajMat[startG][nextG][pathId]
# Get the path data
pathX, pathY, pathT = _path
pathL = trajectory_arclength(_path)
# Total path length
pathSize = len(pathX)

# Prediction of single paths with a mixture model
mgps     = mixtureOfGPs(startG,goalsData)
nSamples = 50

# Divides the trajectory in part_num parts and consider
part_num = 5


# For different sub-parts of the trajectory
for i in range(1,part_num-1):
    p = plotter("imgs/train_station.jpg")
    p.plot_scene_structure(goalsData)

    knownN = int((i+1)*(pathSize/part_num)) #numero de datos conocidos
    observations = observed_data([pathX,pathY,pathL],knownN)
    """Multigoal prediction test"""
    print('[INF] Updating likelihoods')
    likelihoods = mgps.update(observations)
    print('[INF] Performing prediction')
    predictedXYVec,varXYVec = mgps.predict_path()
    print('[INF] Plotting')
    p.plot_multiple_predictions_and_goal_likelihood(pathX,pathY,knownN,goalsData.nGoals,likelihoods,predictedXYVec,varXYVec)
    print("[RES] Goals likelihood\n",mgps.goalsLikelihood)
    print("[RES] Mean likelihood:", mgps.meanLikelihood)
    print('[INF] Generating samples')
    vecX,vecY,__ = mgps.sample_paths(nSamples)
    trueX = observations[:,0]
    trueY = observations[:,1]
    p.plot_path_samples_with_observations(trueX.reshape((-1,1)),trueY.reshape((-1,1)),vecX,vecY)
    p.show()
