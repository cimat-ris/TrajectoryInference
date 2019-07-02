"""
@author: karenlc
"""
from GPRlib import *
from path import *
from testing import *
from plotting import *
from multipleAgents import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Ellipse
from copy import copy

#******************************************************************************#

def single_goal_prediction_test(img,x,y,l,knownN,startG,finishG,goals,unitMat,stepUnit,kernelMatX,kernelMatY,goalSamplingAxis):
    predictedX, predictedY, varX, varY, newL = trajectory_prediction_test(img,x,y,l,knownN,startG,finishG,goals,unitMat,stepUnit,kernelMatX,kernelMatY)
    initTime = knownTime[knownN-1]
    time = arclen_to_time(initTime,newL,speed)
    #print("[newL]:",newL)
    #print("Predictions:\n x:",predictedX,"\ny:",predictedY,"\nl:",newL)
    print("[Arclen to Time]:",time)
    #plot_prediction(img,x,y,knownN,predictedX, predictedY,varX,varY)

def goal_to_subgoal_prediction_error(x,y,l,knownN,startG,finishG,goals,subgoals,unitMat,stepUnit,kernelMatX,kernelMatY,subgoalsUnitMat,subgoalsKernelMatX,subgoalsKernelMatY):
    trueX, trueY, trueL = get_known_set(x,y,l,knownN)
    predictedX, predictedY, varX, varY = trajectory_prediction_test(x,y,l,knownN,startG,finishG,goals,unitMat,stepUnit,kernelMatX,kernelMatY)

    subG = 2*finishG
    predictedX_0, predictedY_0, varX_0, varY_0 = trajectory_prediction_test(x,y,l,knownN,startG,subG,subgoals,subgoalsUnitMat,stepUnit,subgoalsKernelMatX,subgoalsKernelMatY)

    subG = 2*finishG +1
    predictedX_1, predictedY_1, varX_1, varY_1 = trajectory_prediction_test(x,y,l,knownN,startG,subG,subgoals,subgoalsUnitMat,stepUnit,subgoalsKernelMatX,subgoalsKernelMatY)    #plot_prediction(img,x,y,knownN,predictedX, predictedY,varX,varY)
    #errores
    N = len(x)
    realX, realY = [], []
    for i in range(knownN,N):
        realX.append(x[i])
        realY.append(y[i])
    #print("longitudes de vec:",len(realX), len(predictedX))
    error = average_displacement_error([realX,realY],[predictedX,predictedY])
    error0 = average_displacement_error([realX,realY],[predictedX_0,predictedY_0])
    error1 = average_displacement_error([realX,realY],[predictedX_1,predictedY_1])
    return error, error0, error1

"""******************************************************************************"""
"""**************    Main function starts here      *****************************"""

# Areas de interes [x1,y1,x2,y2,...]
#R0 = [400,40,680,40,400,230,680,230] #azul
#R1 = [1110,40,1400,40,1110,230,1400,230] #cian
R0 = [410,10,680,10,410,150,680,150] #azul
R1 = [1120,30,1400,30,1120,150,1400,150] #cian
R2 = [1650,460,1810,460,1650,740,1810,740] #verde
R3 = [1500,950,1800,950,1500,1080,1800,1080]#amarillo
R4 = [100,950,500,950,100,1080,500,1080] #naranja
R5 = [300,210,450,210,300,400,450,400] #rosa
R6 = [0,430,80,430,0,600,80,600]
R7 = [1550,300,1650,300,1550,455,1650,455]
R8 = [500,950,1000,950,500,1080,1000,1080]
R9 = [1000,950,1500,950,1000,1080,1500,1080]
goalSamplingAxis = ['x','x','y','x','x','y','y','y']

# This array will contain the zones of interest
areas = [R0,R1,R2,R3,R4,R5]
nGoals = len(areas)
img = mpimg.imread('imgs/goals.jpg')

# Al leer cortamos las trayectorias multiobjetivos por pares consecutivos
# y las agregamos como trayectorias independientes
dataPaths, multigoal = get_paths_from_file('datasets/CentralStation_paths_10000.txt',areas)
usefulPaths = getUsefulPaths(dataPaths,areas)
sortedPaths = sorted(usefulPaths, key=time_compare)

print("[INF] Number of useful paths: ",len(usefulPaths))

"""
Useful matrices:
- pathMat: Quita las trayectorias que se alejan de la media del conjunto que va de g_i a g_j
- meanLenMat: Guarda en ij el arc-len promedio de los caminos de g_i a g_j
- euclideanDistMat: Guarda en ij la distancia euclidiana del goal g_i al g_j
- unitMat: Guarda en ij la unidad de distancia para los caminos de g_i a g_j
"""
# Split the trajectories into pairs of goals
startToGoalPath, arclenMat = define_trajectories_start_and_end_areas(areas,areas,usefulPaths)

# Remove the trajectories that are either too short or too long
pathMat, learnSet = filter_path_matrix(startToGoalPath, nGoals, nGoals)
#plotPaths(pathMat, img)

print("[INF] Number of filtered paths: ",len(learnSet))

# Compute the mean lengths
meanLenMat = get_mean_length(pathMat, nGoals)
# Compute the distances between pairs of goals (as a nGoalsxnGoals matrix)
euclideanDistMat = get_euclidean_goal_distance(areas, nGoals)
# Compute the ratios between average path lengths and inter-goal distances
unitMat,  meanUnit = get_distance_unit(meanLenMat, euclideanDistMat, nGoals)

stepUnit = 0.0438780780171   #get_number_of_steps_unit(pathMat, nGoals)
speed    = 1.65033755511     #get_pedestrian_average_speed(dataPaths)
# Computer prior probabilities between goals
priorTransitionMat               = prior_probability_matrix(pathMat, nGoals)
# For each pair of goals, determine the line prior
linearPriorMatX, linearPriorMatY = get_linear_prior_mean_matrix(pathMat, nGoals, nGoals)

#print("Linear prior mean Mat X:\n", linearPriorMatX)
#print("Linear prior mean Mat Y:\n", linearPriorMatY)
#print("***arc-len promedio***\n", meanLenMat)
#print("***distancia euclidiana entre goals***\n", euclideanDistMat)
#print("Prior likelihood matrix:", priorLikelihoodMat)
kernelType = "linePriorCombined"#"combined"
nParameters = 4

"""******************************************************************************"""
"""**************    Learning / reading parameters     **************************"""
learningParameters = False
if learningParameters==True:
    print("[INF] Starting the learning phase")
    kernelMat_x, kernelMat_y = optimize_parameters_between_goals(kernelType, pathMat, nGoals, nGoals, linearPriorMatX, linearPriorMatY)
    write_parameters(kernelMat_x,nGoals,nGoals,"linearpriorcombined6x6_x.txt")
    write_parameters(kernelMat_y,nGoals,nGoals,"linearpriorcombined6x6_y.txt")
    print("[INF] End of the learning phase")
else:
     # Read the kernel parameters from file
     kernelMat_x = read_and_set_parameters("linearpriorcombined6x6_x.txt",nParameters)
     kernelMat_y = read_and_set_parameters("linearpriorcombined6x6_y.txt",nParameters)

"""******************************************************************************"""
"""**************    Testing                           **************************"""
# We give the start and ending goals
startG = 1
nextG = 4

# Kernels for this pair of goals
kernelX = kernelMat_x[startG][nextG]
kernelY = kernelMat_y[startG][nextG]

# Index of the trajectory to predict
pathId = 3
# Get the ground truth path
_path = pathMat[startG][nextG][pathId]
# Get the path data
pathX, pathY, pathL, pathT = _path.x, _path.y, _path.l, _path.t
# Total path length
pathSize = len(pathX)

arcLenToTime = arclen_to_time(320,pathL,speed)
#print("[time]",arcLenToTime)
#print("mean error:", mean_error(pathT,arcLenToTime))
#print("[path data]:\n [x]:",pathX,"\n[y]:",pathY,"\n[l]:",pathL,"\n[t]:",pathT)

predictionTest = False
if predictionTest==True:
    # The dataset of observations is split into part_num subsets
    # When predicting, one takes the percentge i/part_num as known,
    # and the second part is predicted
    part_num = 10
    steps = 10
    for i in range(1,part_num-1):
        knownN = int((i+1)*(pathSize/part_num)) #numero de datos conocidos
        trueX,trueY,trueL = get_known_set(pathX,pathY,pathL,knownN)
        """Simple prediction test"""
        knownTime = pathT[0:knownN]
        rT = pathT[knownN:pathSize]
        """Multigoal prediction test"""
        multigoal_prediction_test_lp(img,trueX,trueY,trueL,knownN,startG,areas,unitMat,stepUnit,kernelMat_x,kernelMat_y,priorTransitionMat,linearPriorMatX,linearPriorMatY,goalSamplingAxis)
        #plot_euclidean_distance_to_finish_point(img,trueX,trueY,knownN,middle_of_area(areas[nextG]))
        #prediction_test_over_time(pathX,pathY,pathT,knownN,start[0],nextG[0],areas)

nSamples = 100
startGoal, finishGoal = 0,2
#path_sampling_between_goals_test(img,nSamples,areas,startGoal,finishGoal,goalSamplingAxis,unitMat,stepUnit,kernelMat_x,kernelMat_y,linearPriorMatX,linearPriorMatY)

knownN = int(pathSize/2) #numero de datos conocidos
trueX,trueY,trueL = get_known_set(pathX,pathY,pathL,knownN)
path_sampling_to_goal_test(img,trueX,trueY,trueL,knownN,nSamples,areas,startGoal,finishGoal,goalSamplingAxis,unitMat,stepUnit,kernelMat_x,kernelMat_y,linearPriorMatX,linearPriorMatY)

quit()

#Prueba el error de la prediccion variando:
# - el numero de muestras del punto final
# - numero de pasos a comparar dado un objetivo final
#number_of_samples_and_points_to_compare_to_destination(areas,pathMat,nGoals,nGoals,unitMat,meanLenMat,goalSamplingAxis,kernelMat_x,kernelMat_y)

interactionTest = False
if interactionTest == True:
    sortedSet     = get_path_set_given_time_interval(sortedPaths,0,200)
    print("Numero de trayectorias en el conjunto:",len(sortedSet))
    Ti, Tj = 1,0
    interaction_potential(sortedSet[Ti], sortedSet[Tj])
    plotPathSet([sortedSet[Ti],sortedSet[Tj]  ],img)

#compare_error_goal_to_subgoal_test(img,pathX,pathY,pathL,startG,nextG,areas,unitMat,stepUnit,kernelMat_x,kernelMat_y,goalSamplingAxis)
#path_sampling_test(img,areas,nGoals,goalSamplingAxis,unitMat,stepUnit,kernelMat_x,kernelMat_y,linearPriorMatX,linearPriorMatY)

"""
for i in range(0):#1,nGoals):
    for j in range(nGoals):
        startG, finishG = i,j
        _trajectorySet = pathMat[startG][finishG]
        if(len(_trajectorySet) > 0):
            trajectorySet = []
            _min = min(100, len(_trajectorySet))
            for k in range(_min):
                trajectorySet.append(_trajectorySet[k])
            print("[",i,",",j,"]")
            test_prediction_goal_to_subgoal(trajectorySet,startG,finishG,areas,subgoals,unitMat,stepUnit,kernelMat_x,kernelMat_y,subgoalsUnitMat,subgoalsKernelMat_x,subgoalsKernelMat_y)
"""
