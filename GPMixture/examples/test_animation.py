"""
@author: karenlc
"""
from test_common import *
from gp_code.mixture_gp import mixtureOfGPs

# Read the areas file, dataset, and form the goalsLearnedStructure object
goalsDescriptions= './parameters/CentralStation_GoalsDescriptions.csv'
trajFile         = './datasets/GC/Annotation/'
imgGCS           = './imgs/train_station.jpg'
img = mpimg.imread('imgs/train_station.jpg')

traj_dataset, goalsData, trajMat, __ = read_and_filter('GCS',goalsDescriptions,trajFile,use_pickled_data=True)

# Selection of the kernel type
kernelType  = "linePriorCombined"
nParameters = 4

# Read the kernel parameters from file
goalsData.kernelsX = read_and_set_parameters("parameters/linearpriorcombined20x20_x.txt",nParameters)
goalsData.kernelsY = read_and_set_parameters("parameters/linearpriorcombined20x20_y.txt",nParameters)

"""**********          Testing          ***********"""
# We give the start and ending goals
startG, nextG, pathId = 0,6,7

# Kernels for this pair of goals
kernelX = goalsData.kernelsX[startG][nextG]
kernelY = goalsData.kernelsY[startG][nextG]

# Get the ground truth path
path = trajMat[startG][nextG][pathId]
# Get the path data
pathX, pathY, pathT = path
pathL = trajectory_arclength(path)
# Total path length
pathSize = len(pathX)

# Test function: prediction of single paths with multiple goals
mgps = mixtureOfGPs(startG,goalsData)

part_num = 10
# For different sub-parts of the trajectory
for i in range(1,part_num-1):
    knownN = int((i+1)*(pathSize/part_num)) 
    trueX,trueY,trueL = observed_data(pathX,pathY,pathL,knownN)
    """Multigoal prediction test"""
    print('[INF] Updating likelihoods')
    likelihoods = mgps.update(trueX,trueY,trueL)
    print('[INF] Performing prediction')
    predictedXYVec,varXYVec = mgps.predict()
    print('[INF] Plotting')
    animate_multiple_predictions_and_goal_likelihood(img,pathX,pathY,knownN,goalsData.nGoals,likelihoods,predictedXYVec,varXYVec,False)
    if i==3:
        animate_multiple_predictions_and_goal_likelihood(img,pathX,pathY,knownN,goalsData.nGoals,likelihoods,predictedXYVec,varXYVec,True)
