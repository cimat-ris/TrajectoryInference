"""
@author: karenlc
"""
from test_common import *
from gp_code.mixture_gp import mixtureOfGPs
from gp_code.single_gp import singleGP

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
startG, endG = 0,6
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

# Test function: prediction of single paths with multiple goals
gp = singleGP(startG,endG,goalsData)

p = plotter()
# For different sub-parts of the trajectory
for knownN in range(10,pathSize):
    p.set_background(imgGCS)
    p.plot_scene_structure(goalsData)
    observations, ground_truth = observed_data([pathX,pathY,pathL,pathT],knownN)
    """Single goal prediction test"""
    print('[INF] Updating likelihoods')
    # Update the GP with (real) observations
    start        = time.process_time()
    likelihood   = gp.update(observations)
    stop         = time.process_time()
    filteredPath = gp.filter()
    print("[INF] CPU process time (update): %.1f [ms]" % (1000.0*(stop-start)))
    start = stop
    # Perform prediction
    predictedXY,varXY = gp.predict_trajectory()
    stop       = time.process_time()
    print("[INF] CPU process time (prediction): %.1f [ms]" % (1000.0*(stop-start)))
    print("[RES] Likelihood: ",likelihood)
    print('[INF] Plotting')
    # Plot the filtered version of the observations
    p.plot_filtered(filteredPath)
    # Plot the prediction
    p.plot_prediction(observations,predictedXY,varXY)
    # Plot the ground truth
    p.plot_ground_truth(ground_truth)
    p.pause(0.05)
