"""
@author: karenlc
"""
from test_common import *
from gp_code.single_gp import singleGP
import matplotlib.pyplot as plt

# Read the areas file, dataset, and form the goalsLearnedStructure object
goalsDescriptions= './parameters/CentralStation_GoalsDescriptions.csv'
trajFile         = './datasets/GC/Annotation/'
imgGCS           = './imgs/train_station.jpg'
img              = mpimg.imread('imgs/train_station.jpg')

traj_dataset, goalsData, trajMat, __ = read_and_filter('GCS',goalsDescriptions,trajFile,use_pickled_data=True)

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
pathS = trajectory_speeds(path)
# Total path length
pathSize = len(pathX)

# Test function: prediction of single trajectories with a single goal
gp = singleGP(startG,endG,goalsData)

fig, ax = plt.subplots(3,1)
ax[0].set_ylabel('x')
ax[0].set_xlabel('l')
ax[1].set_ylabel('y')
ax[1].set_xlabel('l')
ax[2].set_ylabel('v')
ax[2].set_xlabel('l')

# For different sub-parts of the trajectory
for knownN in range(10,pathSize-1):
    print('--------------------------')
    # Observations: x,y,l,t
    # Ground truth: x,y,t
    observations, ground_truth = observed_data([pathX,pathY,pathL,pathT],knownN)
    """Single goal prediction test"""
    print('[INF] Updating observations')
    # Update the GP with (real) observations
    likelihood   = gp.update(observations)
    filteredPath = gp.filter()
    # Perform prediction
    predictedXYLT,varXY = gp.predict_trajectory()
    print('[INF] Plotting')
    ax[0].axis([0,np.max(pathL),0,np.max(pathX)])
    ax[0].plot(predictedXYLT[:,2],predictedXYLT[:,0],'b')
    ax[0].plot(observations[:,2],observations[:,0],'r')
    ax[1].axis([0,np.max(pathL),0,np.max(pathY)])
    ax[1].plot(predictedXYLT[:,2],predictedXYLT[:,1],'b')
    ax[1].plot(observations[:,2],observations[:,1],'r')
    ax[2].axis([0,np.max(pathL),0,np.max(pathS)])
    ax[2].plot(predictedXYLT[:,2],predictedXYLT[:,4],'b')
    ax[2].plot(observations[:,2],pathS[:knownN],'r')
    plt.pause(0.15)
    ax[0].cla()
    ax[1].cla()
    ax[2].cla()
