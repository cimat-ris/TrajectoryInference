"""
@author: karenlc
"""
import random
from test_common import *
from gp_code.single_gp import singleGP

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

"""**********          Testing          ***********"""
# We select a pair of starting and ending goals, and a trajectory id
randomPath = False
if randomPath:
    flag = True
    while flag:
        gi, gj = random.randrange(goalsData.nGoals), random.randrange(goalsData.nGoals)
        if len(trajMat[gi][gj]) > 0:
            pathId = random.randrange( len(trajMat[gi][gj]) )
            flag = False
    print("[INF] Selected goals:",(gi,gj),"| path index:", pathId)
else:
    gi, gj, pathId = 0, 6, 5

# Get the ground truth path
path  = trajMat[gi][gj][pathId]
pathX, pathY, pathT = path
pathL = trajectory_arclength(path)
# Total path length
pathSize = len(pathX)

# Prediction of single paths with single goals
gp = singleGP(gi,gj,goalsData)

# Divides the trajectory in part_num parts and infer the posterior over the remaining part
part_num = 10
for i in range(1,part_num-1):
    p = plotter(imgGCS)
    p.plot_scene_structure(goalsData)
    # Data we will suppose known
    knownN = int((i+1)*(pathSize/part_num))
    observations = observed_data([pathX,pathY,pathL,pathT],knownN)
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
    p.plot_filtered(filteredPath[0,:],filteredPath[1,:])
    # Plot the prediction
    p.plot_prediction(pathX,pathY,knownN,predictedXY,varXY)
    p.show()

# Same as above, with samples instead
part_num = 10
for i in range(1,part_num-1):
    p = plotter(imgGCS)
    p.plot_scene_structure(goalsData)
    # Data we will suppose known
    knownN            = int((i+1)*(pathSize/part_num))
    observations = observed_data([pathX,pathY,pathL,pathT],knownN)
    """Single goal prediction test"""
    # Update the GP with (real) observations
    start      = time.process_time()
    likelihood = gp.update(observations)
    stop       = time.process_time()
    print("[INF] CPU process time (update): %.1f [ms]" % (1000.0*(stop-start)))
    start = stop
    # Generate samples
    predictedXY,varXY = gp.predict_path(compute_sqRoot=True)
    vecX,vecY         = gp.sample_paths(10)
    p.plot_path_samples_with_observations(observations[:,0].reshape((-1,1)),observations[:,1].reshape((-1,1)),vecX,vecY)
    stop       = time.process_time()
    print("[INF] CPU process time (sampling): %.1f [ms]" % (1000.0*(stop-start)))
    print('[INF] Plotting')
    print("[RES] [Likelihood]: ",likelihood)
    p.show()
