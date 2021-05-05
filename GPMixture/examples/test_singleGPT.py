"""
Test single GP | Trautman's approach
"""
import random
from test_common import *
from gp_code.single_gp import singleGP
from gp_code.kernels import *

# Read the areas file, dataset, and form the goalsLearnedStructure object
goalsDescriptions= 'parameters/CentralStation_GoalsDescriptions.csv'
trajFile         = 'datasets/GC/Annotation/'
imgGCS           = 'imgs/train_station.jpg'

traj_dataset, goalsData, trajMat, __, __ = read_and_filter('GCS',goalsDescriptions,trajFile,use_pickled_data=True)

#I'm skipping the training for now

goalsData.kernelsX = create_kernel_matrix('combinedTrautman', goalsData.goals_n, goalsData.goals_n)
goalsData.kernelsY = create_kernel_matrix('combinedTrautman', goalsData.goals_n, goalsData.goals_n)

"""
print('[INF] Kernel parameters')
for row in goalsData.kernelsX:
    for ker in row:
        ker.print_parameters()
"""
"""**********          Testing          ***********"""
gi, gj, pathId = 0, 6, 5

# Get the ground truth path
path = trajMat[gi][gj][pathId]
pathX, pathY, pathT = path
# Total path length
pathSize = len(pathX)

# Prediction of single paths with single goals
gp = singleGP(gi,gj,goalsData,'Trautman')

# Divides the trajectory in part_num parts and infer the posterior over the remaining part
part_num = 3
for i in range(1,part_num-1):
    p = plotter(imgGCS)
    p.plot_scene_structure(goalsData)
    # Data we will suppose known
    knownN = int((i+1)*(pathSize/part_num))
    observations = observed_data(path,knownN)
    """Single goal prediction test"""
    # Update the GP with (real) observations
    start               = time.process_time()
    likelihood          = gp.update(observations)
    stop                = time.process_time()
    # Perform prediction
    print('--- prediction---')
    predictedXY,varXY = gp.predict_path()
    print('[INF] Plotting')
    print("[RES] [Likelihood]: ",likelihood)
    # Plot the filtered version of the observations
    #p.plot_filtered(filteredX,filteredY)
    # Plot the prediction
    p.plot_prediction(observations,predictedXY,varXY)
    p.show()
