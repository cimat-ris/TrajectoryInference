"""
@author: karenlc
"""
from test_common import *
from gp_code.mixtureOfGPs import mixtureOfGPs

# Read the areas data from a file and take only the first 6 goals
data     = pd.read_csv('parameters/CentralStation_areasDescriptions.csv')
areas    = data.values[:6,2:]
areasAxis= data.values[:6,1]
nGoals   = len(areas)
img         = mpimg.imread('imgs/goals.jpg')
station_img = mpimg.imread('imgs/train_station.jpg')
# We process here multi-objective trajectories into sub-trajectories
dataPaths, multigoal = get_paths_from_file('datasets/CentralStation_trainingSet.txt',areas)
usefulPaths          = getUsefulPaths(dataPaths,areas)
print("[INF] Number of useful paths: ",len(usefulPaths),"/",len(dataPaths))
# Split the trajectories into pairs of goals
startToGoalPath, arclenMat = define_trajectories_start_and_end_areas(areas,areas,usefulPaths)
# Remove the trajectories that are either too short or too long
pathMat, learnSet = filter_path_matrix(startToGoalPath, nGoals, nGoals)
sortedPaths = sorted(learnSet, key=time_compare)
print("[INF] Number of filtered paths: ",len(learnSet))

# Form the object goalsLearnedStructure
goalsData = goalsLearnedStructure(areas,areasAxis,pathMat)
stepUnit  = 0.0438780780171   #get_number_of_steps_unit(pathMat, nGoals)
speed     = 1.65033755511     #get_pedestrian_average_speed(dataPaths)

# For each pair of goals, determine the line priors
useLinearPriors = True
if useLinearPriors:
    goalsData.compute_linear_priors(pathMat)

# Selection of the kernel type
kernelType = "linePriorCombined"#"combined"
nParameters = 4

# Read the kernel parameters from file
goalsData.kernelsX = read_and_set_parameters("parameters/linearpriorcombined6x6_x.txt",nParameters)
goalsData.kernelsY = read_and_set_parameters("parameters/linearpriorcombined6x6_y.txt",nParameters)

"""******************************************************************************"""
"""**************    Testing                           **************************"""
# We give the start and ending goals
startG = 0
nextG = 2

# Kernels for this pair of goals
kernelX = goalsData.kernelsX[startG][nextG]
kernelY = goalsData.kernelsY[startG][nextG]

# Index of the trajectory to predict
pathId = 3
# Get the ground truth path
_path = pathMat[startG][nextG][pathId]
# Get the path data
pathX, pathY, pathL, pathT = _path.x, _path.y, _path.l, _path.t
# Total path length
pathSize = len(pathX)

# Prediction of single paths with a mixture model
mgps     = mixtureOfGPs(startG,stepUnit,goalsData)
nSamples = 50

# Divides the trajectory in part_num parts and consider
part_num = 5

# For different sub-parts of the trajectory
for i in range(1,part_num-1):
    knownN = int((i+1)*(pathSize/part_num)) #numero de datos conocidos
    trueX,trueY,trueL = get_known_set(pathX,pathY,pathL,knownN)
    """Multigoal prediction test"""
    print('[INF] Updating likelihoods')
    likelihoods = mgps.update(trueX,trueY,trueL)
    print('[INF] Performing prediction')
    predictedXYVec,varXYVec = mgps.predict()
    print('[INF] Plotting')
    plot_multiple_predictions_and_goal_likelihood(img,pathX,pathY,knownN,goalsData.nGoals,likelihoods,predictedXYVec,varXYVec)
    print("[RES] Goals likelihood\n",mgps.goalsLikelihood)
    print("[RES] Mean likelihood:", mgps.meanLikelihood)
    print('[INF] Generating samples')
    vecX,vecY,__ = mgps.generate_samples(nSamples)
    plot_path_samples_with_observations(img,trueX,trueY,vecX,vecY)
