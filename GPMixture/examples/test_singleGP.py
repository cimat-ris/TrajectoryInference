"""
@author: karenlc
"""
from test_common import *
from gp_code.single_gp import singleGP

# Read the areas file, dataset, and form the goalsLearnedStructure object
goalsDescriptions= './parameters/CentralStation_GoalsDescriptions.csv'
trajFile         = './datasets/GC/Annotation/'
GCSimg           = './imgs/train_station.jpg'

traj_dataset,goalsData, trajMat, __ = read_and_filter_('GCS',goalsDescriptions,trajFile,use_pickled_data=True)
# TODO: we should remove this parameter; a priori it could be deduced in some way with the speed
stepUnit  = 0.0438780780171   #get_number_of_steps_unit(pathMat, nGoals)

# Selection of the kernel type
kernelType = "linePriorCombined"
nParameters = 4
# Read the kernel parameters from file
goalsData.kernelsX = read_and_set_parameters("parameters/linearpriorcombined20x20_x.txt",nParameters)
goalsData.kernelsY = read_and_set_parameters("parameters/linearpriorcombined20x20_y.txt",nParameters)

"""**************    Testing           **************************"""
# We select a pair of starting and ending goals, and a trajectory id
startG = 0
nextG  = 6
pathId = 3

# Kernels for this pair of goals
kernelX = goalsData.kernelsX[startG][nextG]
kernelY = goalsData.kernelsY[startG][nextG]
# Get the ground truth path
pathX, pathY, pathT = trajMat[startG][nextG][pathId]
# Total path length
pathSize = len(pathX)

# Prediction of single paths with single goals
gp = singleGP(startG,nextG,stepUnit,goalsData)

# Divides the trajectory in part_num parts and consider
part_num = 10
for i in range(1,part_num-1):
    p = plotter("imgs/train_station.jpg")
    p.plot_scene_structure(goalsData)
    # Data we will suppose known
    knownN = int((i+1)*(pathSize/part_num))
    trueX,trueY,trueL = observed_data(pathX,pathY,pathT,knownN)
    """Single goal prediction test"""
    # Update the GP with (real) observations
    start      = time.process_time()
    likelihood = gp.update(trueX,trueY,trueL)
    stop       = time.process_time()
    print("[INF] CPU process time (update): %.1f [ms]" % (1000.0*(stop-start)))
    start = stop
    # Perform prediction
    predictedXY,varXY = gp.predict()
    stop       = time.process_time()
    print("[INF] CPU process time (prediction): %.1f [ms]" % (1000.0*(stop-start)))
    print('[INF] Plotting')
    print("[RES] [Likelihood]: ",likelihood)
    # Generate samples
    vecX,vecY         = gp.generate_samples(10)
    p.plot_path_samples_with_observations(trueX,trueY,vecX,vecY)
    p.plot_prediction(pathX,pathY,knownN,predictedXY,varXY)
    p.show()
