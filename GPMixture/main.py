"""
@author: karenlc
"""
from GPRlib import *
from path import *
from testing import *
from plotting import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Ellipse
from copy import copy
    
#******************************************************************************#
# Lectura de los nombres de los archivos de datos
def readDataset(name):   
    file = open(name,'r')
    lines = file.readlines()
    for i in range(len(lines)):
        lines[i] = lines[i].strip("\n")
    return lines

#Regresa una matriz con la probabilidad de ir de g_i a g_j en cada entrada
def next_goal_probability_matrix(M, nGoals):
    probMat = []
    for i in range(nGoals):
        p = []
        n = 0.
        for j in range(nGoals):
            n += len(M[i][j])
        
        for j in range(nGoals):
            if n == 0:
                p.append(0.)
            else:
                p.append(float(len(M[i][j])/n))
        probMat.append(p)
        
    return probMat
    
def getKnownData(x,y,z,percent):
    trueX, trueY, trueZ = [],[],[]
    for j in range(len(x)):
        M = int(len(x[j])*percent)  
        if M == 0:
            return [],[],[]
        auxX, auxY, auxZ = np.zeros(M), np.zeros(M), np.zeros(M)
        for i in range(M):
            auxX[i] = x[j][i]
            auxY[i] = y[j][i]
            auxZ[i] = z[j][i]
        if len(x[j]) > 0:
            auxX[M-1] = x[j][len(x[j])-1]
            auxY[M-1] = y[j][len(y[j])-1]
            auxZ[M-1] = z[j][len(z[j])-1]

        trueX.append(auxX)
        trueY.append(auxY)
        trueZ.append(auxZ)
    
    return trueX, trueY, trueZ
         

def most_likely_goals(likelihood, nGoals):
    next_goals = []
    likely = copy(likelihood)
    for i in range(2):
        maxVal = 0
        maxInd = 0
        for j in range(nGoals):
            if likely[j] > maxVal:
                maxVal = likely[j]
                maxInd = j
        next_goals.append(maxInd)
        likely[maxInd] = 0
    return next_goals
    
def get_number_of_steps_unit(Mat, nGoals):
    unit = 0.0
    numUnits = 0
    for i in range(nGoals):
        for j in range(nGoals):
            numPaths = len(Mat[i][j])
            meanU = 0.0
            for k in range(numPaths):
                path = Mat[i][j][k]
                l = path.l[len(path.l)-1]
                if(l == 0):
                    numPaths -= 1
                else: 
                    stps = len(path.l)
                    u = stps/l
                    meanU += u
            if(numPaths > 0):
                meanU = meanU/numPaths
            if(meanU >0):
                unit += meanU
                numUnits += 1
    unit = unit/numUnits
    #print("mean mean unit:", unit)
    return unit
    
def copy_unitMat(unitMat, nGoals, nSubgoals):
    mat = []
    m = int(nSubgoals/nGoals)
    for i in range(nGoals):
        r = []
        for j in range(nSubgoals):
            k = int(j/m)
            r.append(unitMat[i][k])
        mat.append(r)
    return mat
        
#******************************************************************************#
    
def single_goal_prediction_test(x,y,l,knownN,startG,finishG,goals,unitMat,stepUnit,kernelMatX,kernelMatY,goalSamplingAxis): 
    predictedX, predictedY, varX, varY = trajectory_prediction_test(x,y,l,knownN,startG,finishG,goals,unitMat,stepUnit,kernelMatX,kernelMatY) 
    lenX = goals[finishG][len(goals[finishG]) -2] - goals[finishG][0]
    lenY = goals[finishG][len(goals[finishG]) -1] - goals[finishG][1]
    elipse = [lenX, lenY]
    
    plot_prediction(img,x,y,knownN,predictedX, predictedY,varX,varY,elipse)
    
def get_error_of_final_point_comparing_k_steps(k,trajectorySet,startG,finishG,goals,unitMat,meanLenMat):
    kernelX = kernelMat_x[startG][finishG]
    kernelY = kernelMat_y[startG][finishG]

    meanError = 0.0    
    for i in range( len(trajectorySet) ):
        x = trajectorySet[i].x
        y = trajectorySet[i].y
        l = trajectorySet[i].l
        final = [x[len(x)-1], y[len(y)-1]]
        knownN = len(x)* (0.9)
        trueX, trueY, trueZ = get_known_set(x,y,l,knownN)
        #lastKnownPoint = [x[knownN-1], y[knownN-1], l[knownN-1] ]
        unit = unitMat[startG][finishG]
        
        predicted_final = get_finish_point(k,trueX,trueY,trueZ,finishG,goals,kernelX,kernelY,unit,img,goalSamplingAxis)
        error = final_displacement_error(final,predicted_final)
        #print("FDE: ",error)
        meanError += error
    meanError /= len(trajectorySet)
    return meanError
    
def choose_number_of_steps_and_samples(trajectorySet,startG,finishG,goals,unitMat,meanLenMat):
    errorVec = []    
    stepsVec = []
    samplesVec = []
    steps = 1
    samples = 1
    for i in range(20):
        samplesVec.append(samples)
        error = get_error_of_final_point_comparing_k_steps(samples,trajectorySet,startG,finishG,goals,unitMat,meanLenMat)
        errorVec.append(error)
        samples += 1
        
    print("error: ", errorVec)
    plt.plot(samplesVec, errorVec)
    plt.xlabel('size of sample')
    plt.ylabel('mean FDE')
    #para ambos el error sera FDE
    #en 2 te detienes cuando el error deje de disminuir significativamente
    
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
# Areas de interes [x1,y1,x2,y2,...]
#R0 = [400,40,680,40,400,230,680,230] #azul
#R1 = [1110,40,1400,40,1110,230,1400,230] #cian
R0 = [400,10,680,10,400,150,680,150] #azul
R1 = [1100,10,1400,10,1100,150,1400,150] #cian
R2 = [1650,490,1810,490,1650,740,1810,740] #verde
R3 = [1450,950,1800,950,1450,1080,1800,1080]#amarillo
R4 = [100,950,500,950,100,1080,500,1080] #naranja
R5 = [300,210,450,210,300,400,450,400] #rosa
goalSamplingAxis = ['x','x','y','x','x','y']
#R6 = [750,460,1050,460,750,730,1050,730] #rojo
#Arreglo que contiene las areas de interes
areas = [R0,R1,R2,R3,R4,R5]
nGoals = len(areas)
"""
subgoals =[r00,r01,r02,r03, r10,r11,r12,r13, r20,r21,r22,r23, r30,r31,r32,r33, r40,r41,r42,r43, r50,r51,r52,r53] #[r00,r01,r10,r11,r20,r21,r30,r31,r40,r41,r50,r51]
nSubgoals = len(subgoals)
subgoalSamplingAxis = []
for i in range(nGoals):
    for j in range(4):
        subgoalSamplingAxis.append(goalSamplingAxis[i])
#print("subG sampling axis:", subgoalSamplingAxis)
"""
img = mpimg.imread('imgs/goals.jpg')  

#Al leer cortamos las trayectorias multiobjetivos por pares consecutivos 
#y las agregamos como trayectorias independientes 
true_paths, multigoal = get_paths_from_file('datasets/CentralStation_paths_10000.txt',areas)
usefulPaths = getUsefulPaths(true_paths,areas)
#plotPaths(usefulPaths, img)
#print("useful paths: ",len(usefulPaths))
#Histograma    
#histogram(true_paths,"duration")
"""
Matrices útiles:
- pathMat: Quita las trayectorias que se alejan de la media del conjunto que va de g_i a g_j
- meanLenMat: Guarda en ij el arc-len promedio de los caminos de g_i a g_j
- euclideanDistMat: Guarda en ij la distancia euclidiana del goal g_i al g_j
- unitMat: Guarda en ij la unidad de distancia para los caminos de g_i a g_j
"""
#Separamos las trayectorias en pares de goals
startToGoalPath, arclenMat = define_trajectories_start_and_end_areas(areas,areas,usefulPaths)#true_paths)#learnSet)#
#Quitamos las trayectorias demasiado largas o demasiado cortas
pathMat, learnSet = filter_path_matrix(startToGoalPath, nGoals, nGoals)
meanLenMat = get_mean_length(pathMat, nGoals)
euclideanDistMat = get_euclidean_goal_distance(areas, nGoals)
unitMat = get_unit(meanLenMat, euclideanDistMat, nGoals)
stepUnit = 0.0438780780171 #get_number_of_steps_unit(pathMat, nGoals)
#print("***arc-len promedio***\n", meanLenMat)
#print("***distancia euclidiana entre goals***\n", euclideanDistMat)
#print("***unidades de distancia***\n", unitMat)
priorLikelihoodMat = next_goal_probability_matrix(pathMat, nGoals)
#print("likelihood mat:", priorLikelihoodMat)
"""
#***********APRENDIZAJE***********
print("***********INICIO APRENDIZAJE*********")
kernelMat, parametersMat = create_kernel_matrix("combined", nGoals, nGoals)
kernelMat_x, kernelMat_y = optimize_parameters_between_goals(pathMat, parametersMat, nGoals, nGoals)
write_parameters(kernelMat_x,nGoals,nGoals,"parameters_x.txt")
write_parameters(kernelMat_y,nGoals,nGoals,"parameters_y.txt")
print("***********FIN DEL APRENDIZAJE*********")
"""
#fijamos los parámetros para cada matriz de kernel
kernelMat_x = read_and_set_parameters("parameters_x.txt",nGoals,nGoals,2)
kernelMat_y = read_and_set_parameters("parameters_y.txt",nGoals,nGoals,2)

"""***********TRAYECTORIAS SUBGOALS***********"""
"""
subgoalStartToGoalPath, subgoalsArclenMat = define_trajectories_start_and_end_areas(areas,subgoals,usefulPaths)
subgoalsPathMat, sublearnSet = filter_path_matrix(subgoalStartToGoalPath, nGoals, nSubgoals)

#***********APRENDIZAJE SUBGOALS***********
print("***********INICIO APRENDIZAJE*********")
subgoalsKernelMat, subgoalsParametersMat = create_kernel_matrix("combined", nGoals, nSubgoals)
subgoalsKernelMatX, subgoalsKernelMatY = optimize_parameters_between_goals(subgoalsPathMat, subgoalsParametersMat, nGoals, nSubgoals)
write_parameters(subgoalsKernelMatX,nGoals,nSubgoals,"subgoalsParameters_x.txt")
write_parameters(subgoalsKernelMatY,nGoals,nSubgoals,"subgoalsParameters_y.txt")
print("***********FIN DEL APRENDIZAJE*********")

#fijamos los parámetros para cada matriz de kernel
subgoalsKernelMat_x = read_and_set_parameters("subgoalsParameters_x.txt",nGoals,nSubgoals,2)
subgoalsKernelMat_y = read_and_set_parameters("subgoalsParameters_y.txt",nGoals,nSubgoals,2)
subgoalsUnitMat = copy_unitMat(unitMat, nGoals, nSubgoals); 
"""
#Test dado el goal de inicio y fin
startG = 0
nextG = 2
kernelX = kernelMat_x[startG][nextG]
kernelY = kernelMat_y[startG][nextG]

traj_id = 0 #indice de la trayectoria a predecir
traj = pathMat[startG][nextG][traj_id]
traj_len = len(traj.x) #total de observaciones de la trayectoria
traj_arclen = traj.length #arreglo de la longitud de arco correspondiente a las observaciones 
#likelihoodVector, error_vector = [], []
arclen_vec = []

#El conjunto de datos observados se parte en part_num
#al hacer la prediccion se toma el porcentaje k/part_num como datos conocidos y se predice el resto
part_num = 8
steps = 10

for i in range(3,part_num-1):
    arclen_vec.append( (i+1)*(traj_arclen/float(part_num))  )
    knownN = int((i+1)*(traj_len/part_num)) #numero de datos conocidos
    #print("num de datos conocidos:", knownN)
    #trueX,trueY,trueL = get_known_set(traj.x,traj.y,traj.l,knownN)
    #likelihood, error = get_goals_likelihood(trueX,trueY,trueL,startG,kernelMat_x,kernelMat_x,areas,nGoals)
    #likelihoodVector.append(likelihood)  
    #error_vector.append(error)
    #likely_goals = most_likely_goals(likelihood, nGoals)
    """Simple prediction test"""
    #single_goal_prediction_test(traj.x,traj.y,traj.l,knownN,startG,nextG,areas,unitMat,stepUnit,kernelMat_x,kernelMat_y,goalSamplingAxis)
    trajectory_subgoal_prediction_test(img,traj.x,traj.y,traj.l,knownN,startG,nextG,areas,unitMat,stepUnit,kernelMat_x,kernelMat_y,goalSamplingAxis)
    #prediction_test(img,traj.x,traj.y,traj.l,knownN,startG,nextG,areas,unitMat,meanLenMat,steps,kernelMat_x,kernelMat_y)
    """Multigoal prediction test"""    
    #multigoal_prediction_test(traj.x,traj.y,traj.l,knownN,startG,areas,unitMat,stepUnit,kernelMat_x,kernelMat_y,priorLikelihoodMat)
    #prediction_test_over_time(traj.x,traj.y,traj.t,knownN,start[0],nextG[0],areas)

#compare_error_goal_to_subgoal_test(img,traj.x,traj.y,traj.l,startG,nextG,areas,unitMat,stepUnit,kernelMat_x,kernelMat_y,goalSamplingAxis)


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

"""
#test para encontrar los valores de k y m al elegir el punto final
trajectorySet = []
startGoal, finishGoal = 4,0
#x_, y_, flag = uniform_sampling_1D(15, areas[finishGoal])
#print("_x=",x_)
#print("_y=",y_)
#plotPaths(pathMat[startGoal][finishGoal], img) 
numOfPaths = len(pathMat[startGoal][finishGoal])
lenTrainingSet = int(numOfPaths/2)
if(lenTrainingSet > 50):
    lenTrainingSet = 50
#print( len(pathMat[startGoal][finalGoal]) )
for i in range(lenTrainingSet):
    trajectorySet.append(pathMat[startGoal][finishGoal][i])
#choose_number_of_steps_and_samples(trajectorySet,startGoal,finishGoal,areas,unitMat,meanLenMat)

kVec = []
mVec = []
for i in range(20):
    mVec.append(i+1)
    kVec.append(i+1)
"""
 
 
 
 
 
 
 
 
 