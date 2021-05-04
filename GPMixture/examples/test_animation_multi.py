"""
@author: karenlc
"""
from test_common import *
from gp_code.mixture_gp import mixtureOfGPs
from gp_code.single_gp import singleGP

# Read the areas file, dataset, and form the goalsLearnedStructure object
trajFile         = './datasets/GC/'
imgGCS           = './imgs/train_station.jpg'
img = mpimg.imread('imgs/train_station.jpg')

traj_dataset, goalsData, trajMat, __ = read_and_filter('GCS',trajFile,use_pickled_data=True)

# Selection of the kernel type
kernelType  = "linePriorCombined"
nParameters = 4

# Read the kernel parameters from file
goalsData.kernelsX = read_and_set_parameters("parameters/linearpriorcombined20x20_x.txt",nParameters)
goalsData.kernelsY = read_and_set_parameters("parameters/linearpriorcombined20x20_y.txt",nParameters)

"""**********          Testing          ***********"""
# We give the start and ending goals
randomPath = True
if randomPath:
    flag = True
    while flag:
        startG, endG = random.randrange(goalsData.goals_n), random.randrange(goalsData.goals_n)
        if len(trajMat[startG][endG])>0 and goalsData.kernelsX[startG][endG].optimized is True:
            pathId = random.randrange( len(trajMat[startG][endG]) )
            flag = False
    print("[INF] Selected goals:",(startG,endG),"| path index:", pathId)
else:
    startG,endG = 0, 7
    pathId       = np.random.randint(0,len(trajMat[startG][endG]))
# Kernels for this pair of goals
kernelX = goalsData.kernelsX[startG][endG]
kernelY = goalsData.kernelsY[startG][endG]
# Get the ground truth path
path = trajMat[startG][endG][pathId]
# Get the path data
pathX, pathY, pathT = path
pathL = trajectory_arclength(path)

# Total path length
pathSize = len(pathX)

# Prediction of single paths with a mixture model
mgps     = mixtureOfGPs(startG,goalsData)

p = plotter()
# For different sub-parts of the trajectory
for knownN in range(5,pathSize-1):
    print('--------------------------')
    p.set_background(imgGCS)
    p.plot_scene_structure(goalsData)
    observations, ground_truth = observed_data([pathX,pathY,pathL,pathT],knownN)
    """Single goal prediction test"""
    print('[INF] Updating observations')
    # Update the GP with (real) observations
    likelihoods  = mgps.update(observations)
    filteredPaths= mgps.filter()
    # Perform prediction
    predictedXYVec,varXYVec = mgps.predict_trajectory()
    print('[INF] Plotting')
    p.plot_multiple_predictions_and_goal_likelihood(observations,predictedXYVec,varXYVec,likelihoods)
    print("[RES] Goals likelihood\n",mgps.goalsLikelihood)
    print("[RES] Mean likelihood:", mgps.meanLikelihood)
    # Plot the ground truth
    p.plot_ground_truth(ground_truth)
    p.pause(0.05)
