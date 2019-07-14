"""
@author: karenlc
"""
import path
from gp_code.io_parameters import *
from testing import *
from utils.plotting import *
from mixtureOfGPs import *
from singleGP import *
from gpRegressor import *
from goalsLearnedStructure import *
from multipleAgents import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Ellipse
from copy import copy


# Interest areas [x1,y1,x2,y2,...]
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

# This array will contain the zones of interest
areas     = [R0,R1,R2,R3,R4,R5]
areasAxis = ['x','x','y','x','x','y','y','y']
nGoals    = len(areas)
img       = mpimg.imread('imgs/goals.jpg')

# Al leer cortamos las trayectorias multiobjetivos por pares consecutivos
# y las agregamos como trayectorias independientes
dataPaths, multigoal = get_paths_from_file('datasets/CentralStation_paths_10000.txt',areas)
usefulPaths = getUsefulPaths(dataPaths,areas)
sortedPaths = sorted(usefulPaths, key=time_compare)

print("[INF] Number of useful paths: ",len(usefulPaths))

# Split the trajectories into pairs of goals
startToGoalPath, arclenMat = define_trajectories_start_and_end_areas(areas,areas,usefulPaths)

# Remove the trajectories that are either too short or too long
pathMat, learnSet = filter_path_matrix(startToGoalPath, nGoals, nGoals)
showDataset = False
if showDataset:
    plotPaths(pathMat, img)
print("[INF] Number of filtered paths: ",len(learnSet))

# Form the object goalsLearnedStructure
goalsData = goalsLearnedStructure(areas,areasAxis,pathMat)

stepUnit = 0.0438780780171   #get_number_of_steps_unit(pathMat, nGoals)
speed    = 1.65033755511     #get_pedestrian_average_speed(dataPaths)

# For each pair of goals, determine the line priors
useLinearPriors = True
if useLinearPriors:
    goalsData.compute_linear_priors(pathMat)

# Selection of the kernel type
kernelType = "linePriorCombined"#"combined"
nParameters = 4

"""******************************************************************************"""
"""**************    Learning / reading parameters     **************************"""
learningParameters = False
if learningParameters==True:
    print("[INF] Starting the learning phase")
    goalsData.optimize_kernel_parameters(kernelType, pathMat)
    write_parameters(kernelMat_x,nGoals,nGoals,"linearpriorcombined6x6_x.txt")
    write_parameters(kernelMat_y,nGoals,nGoals,"linearpriorcombined6x6_y.txt")
    print("[INF] End of the learning phase")
else:
     # Read the kernel parameters from file
     goalsData.kernelsX = read_and_set_parameters("linearpriorcombined6x6_x.txt",nParameters)
     goalsData.kernelsY = read_and_set_parameters("linearpriorcombined6x6_y.txt",nParameters)

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

samplingViz = False
if samplingViz==True:
    path_sampling_test(img,stepUnit,goalsData)

singleTest = False
if singleTest==True:
    gp = singleGP(startG,nextG,stepUnit,goalsData)
    part_num = 10
    steps    = 10
    for i in range(1,part_num-1):
        # Data we will suppose known
        knownN = int((i+1)*(pathSize/part_num))
        trueX,trueY,trueL = get_known_set(pathX,pathY,pathL,knownN)
        """Single goal prediction test"""
        # Update the GP
        likelihood        = gp.update(trueX,trueY,trueL)
        # Perform prediction
        predictedXY,varXY = gp.predict()
        plot_prediction(img,pathX,pathY,knownN,predictedXY,varXY)
        print('[INF] Plotting')
        print("[RES] [Likelihood]: ",likelihood)
        # Generate samples
        vecX,vecY         = gp.generate_samples(100)
        plot_path_samples_with_observations(img,trueX,trueY,vecX,vecY)

mixtureTest = False
if mixtureTest==True:
    mgps = mixtureOfGPs(startG,stepUnit,goalsData)
    part_num = 10
    steps = 10
    for i in range(1,part_num-1):
        knownN = int((i+1)*(pathSize/part_num)) #numero de datos conocidos
        trueX,trueY,trueL = get_known_set(pathX,pathY,pathL,knownN)
        """Multigoal prediction test"""
        likelihoods = mgps.update(trueX,trueY,trueL)
        predictedXYVec,varXYVec = mgps.predict()
        print('[INF] Plotting')
        plot_multiple_predictions_and_goal_likelihood(img,pathX,pathY,knownN,goalsData.nGoals,likelihoods,predictedXYVec,varXYVec)
        print("[RES] [Goals likelihood]\n",mgps.goalsLikelihood)
        print("[RES] [Mean likelihood]:", mgps.meanLikelihood)
        vecX,vecY = mgps.generate_samples(100)
        plot_path_samples_with_observations(img,trueX,trueY,vecX,vecY)

#quit()

interactionTest = False
if interactionTest == True:
    sortedSet     = get_path_set_given_time_interval(sortedPaths,300,700)
    print("Numero de trayectorias en el conjunto:",len(sortedSet))
    plotPathSet(sortedSet,img)

    interaction_potential_for_a_set_of_pedestrians(sortedSet)

interactionWithSamplingTest = True
if interactionWithSamplingTest == True:
    sortedSet = get_path_set_given_time_interval(sortedPaths,100,350)
    plotPathSet(sortedSet,img)
    part_num = 3
    for i in range(1):
        samplePathSet = []
        for j in range(len(sortedSet)):
            print("path #",j)
            currentPath = sortedSet[j]
            pathSize = len(currentPath.x)
            knownN = int((i+1)*(pathSize/part_num))
            observedPath = get_partial_path(currentPath,knownN)
            trueX,trueY,trueL = get_known_set(currentPath.x,currentPath.y,currentPath.l,knownN)
            
            startG = get_path_start_goal(observedPath,areas)
            print("[INF] Start goal", startG)
            mgps = mixtureOfGPs(startG,stepUnit,goalsData)
            likelihoods = mgps.update(trueX,trueY,trueL)
            
            predictedXYVec,varXYVec = mgps.predict()#esta linea es para que se llame a gpRegressor.update
            nSamples = 1       
            vecX,vecY = mgps.generate_samples(nSamples)
            sampleXVec, sampleYVec = [], []
            for k in range(nSamples): #num de samples
                x, y = vecX[k], vecY[k]
                sampleX, sampleY = np.reshape(x,(x.shape[0])), np.reshape(y,(y.shape[0]))
                sampleXVec.append(sampleX)
                sampleYVec.append(sampleY)
                finishG = get_goal_of_point([sampleX[len(sampleX)-1],sampleY[len(sampleY)-1]],areas)
                newL = mgps.gpPathRegressor[finishG].newL
                samplePath = get_path_from_data(observedPath,sampleX,sampleY,newL,speed)   
                samplePathSet.append(samplePath)
        plotPathSet(samplePathSet,img)
        interaction_potential_for_a_set_of_pedestrians(samplePathSet)

#Prueba el error de la prediccion variando:
# - el numero de muestras del punto final
# - numero de pasos a comparar dado un objetivo final
#number_of_samples_and_points_to_compare_to_destination(areas,pathMat,nGoals,nGoals,unitMat,meanLenMat,goalSamplingAxis,kernelMat_x,kernelMat_y)

#Compara el error de la prediccion hacia el centro del goal contra la prediccion hacia los subgoales
#compare_error_goal_to_subgoal_test(img,pathX,pathY,pathL,startG,nextG,areas,unitMat,stepUnit,kernelMat_x,kernelMat_y,goalSamplingAxis)

#path_sampling_test(img,areas,nGoals,goalSamplingAxis,unitMat,stepUnit,kernelMat_x,kernelMat_y,linearPriorMatX,linearPriorMatY)
