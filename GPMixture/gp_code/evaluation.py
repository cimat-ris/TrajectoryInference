"""
Error and likelihood evaluation
"""
import numpy as np
import math
from numpy import linalg as la
from gp_code.regression import *
from utils.manip_trajectories import goal_center_and_size

D = 150.

# Mean euclidean distance between true and predicted data
def mean_euc_error(u,v):
    error = 0.
    for i in range(len(u)):
        error += math.sqrt((u[i]- v[i])**2)
    return error/len(u)

# Mean absolute error (mx,my) between true and predicted data
def mean_abs_error(trueX, trueY, predX, predY):
    e = [0,0]
    lp, l = len(predX), len(trueX)
    for i in range(lp):
        e[0] += abs(trueX[l-1-i]-predX[lp-1-i])
        e[1] += abs(trueY[l-1-i]-predY[lp-1-i])
    e[0] = e[0]/len(predX)
    e[1] = e[1]/len(predY)
    return e

# Average L2 distance between ground truth and our prediction
def mean_displacement_error(true_XY, prediction_XY):
    error = 0.
    trueX, trueY = true_XY[0], true_XY[1]
    predictionX, predictionY = prediction_XY[0], prediction_XY[1]
    l = min(len(trueX),len(predictionX))
    for i in range(l):
        error += math.sqrt((trueX[i]-predictionX[i])**2 + (trueY[i]-predictionY[i])**2)
    if(l>0):
        error = error/l

    return error

#The distance between the predicted final destination and the true final destination
def final_displacement_error(final, predicted_final):
    error = math.sqrt((final[0]-predicted_final[0])**2 + (final[1]-predicted_final[1])**2)
    return error

# Compute the goal likelihood
def compute_goal_likelihood(observedX,observedY,observedL,startG,finishG,stepsToCompare,goalsData):

    error = compute_prediction_error_of_points_along_the_path(stepsToCompare,observedX,observedY,observedL,startG,finishG,goalsData)
    val = goalsData.priorTransitions[startG][finishG]*(math.exp(-1.*( error**2)/D**2 ))#   *(1.-errorG[i])

    return val

# Evaluate the prediction error
def compute_prediction_error_1D(trueX, trueY, prediction, flag):
    error = 0.0
    for i in range(len(prediction) ):
        if flag == 0:
            error += abs(trueX[i] - prediction[i])
        if flag == 1:
            error += abs(trueY[i] - prediction[i])
    return error

# For a given dataset (knownX,knownY,knownL), takes half of the data as known
# and predicts the remaining half. Then, evaluate the prediction error.
def compute_prediction_error_of_points_along_the_path(nPoints,observedX,observedY,observedL,startG,finishG,goalsData):
    # Known data
    observedN = len(observedX)
    halfN     = int(observedN/2)

    # First half of the known data
    trueX = observedX[0:halfN]
    trueY = observedY[0:halfN]
    trueL = observedL[0:halfN]

    # Get the last point and add it to the observed data
    finishXY,__ = goal_center_and_size(goalsData.areas[finishG])
    finishD     = euclidean_distance([trueX[len(trueX)-1],trueY[len(trueY)-1]],finishXY)
    trueX.append(finishXY[0])
    trueY.append(finishXY[1])
    trueL.append(finishD*goalsData.units[startG][finishG])

    d = int(halfN/nPoints)
    realX, realY, predictionSet = [],[],[]
    # Prepare the ground truths and the list of l to evaluate
    for i in range(nPoints):
        realX.append(observedX[halfN + i*d])
        realY.append(observedY[halfN + i*d])
        predictionSet.append(observedL[halfN + i*d])
    # Get the prediction based on the
    predX, predY, varX,varY = prediction_xy(trueX,trueY,trueL, predictionSet, goalsData.kernelsX[startG][finishG],goalsData.kernelsY[startG][finishG],
    goalsData.linearPriorsX[startG][finishG],goalsData.linearPriorsY[startG][finishG])

    # Evaluate the error
    return mean_displacement_error([realX,realY],[predX,predY])

#Toma N-nPoints como datos conocidos y predice los ultimos nPoints, regresa el error de la prediccion
def compute_prediction_error_of_last_known_points(nPoints,knownX,knownY,knownL,goal,unit,stepUnit,kernelX,kernelY):
    knownN = len(knownX)
    trueX = knownX[0:knownN -nPoints]
    trueY = knownY[0:knownN -nPoints]
    trueL = knownL[0:knownN -nPoints]

    finishXY,__ = middle_of_area(goal)
    finishD     = euclidean_distance([trueX[len(trueX)-1],trueY[len(trueY)-1]],finishXY)
    trueX.append(finishXY[0])
    trueY.append(finishXY[1])
    trueL.append(finishD*unit)

    lastX = knownX[knownN -nPoints: nPoints]
    lastY = knownY[knownN -nPoints: nPoints]
    predictionSet = knownL[knownN -nPoints: nPoints]

    predX, predY, varX,varY = prediction_xy(trueX,trueY,trueL, predictionSet, kernelX, kernelY)
    #print("[Prediccion]\n",predX)
    #print(predY)
    error = average_displacement_error([lastX,lastY],[predX,predY])
    #print("[Error]:",error)
    return error

#Busco un alpha en [0,1] tal que t = alpha*T1 + (1-alpha)*T2
def search_value(a, b, t, T1, T2):
    alpha = (a+b)/2
    val = (1-alpha)*T1 + alpha*T2
    if(abs(t-val) < timeRange):
        return alpha
    elif val > t:
        return search_value(a,alpha,t,T1,T2)
    elif val < t:
        return search_value(alpha,b,t,T1,T2)

#Dado un valor en [0,1], regresa un punto (x,y) con x en [path.x_i-1, pathx_i]... usando interpolacion lineal
def get_approximation(val,path,index):
    _x = (1-val)*path.x[index-1] + val*path.x[index]
    _y = (1-val)*path.y[index-1] + val*path.y[index]
    return _x,_y

def ADE_given_future_steps(fullPath, predictedXY, knownN, futureSteps):
    realX = fullPath.x[knownN : knownN+futureSteps]
    realY = fullPath.y[knownN : knownN+futureSteps]

    predX = predictedXY[0][:futureSteps]
    predY = predictedXY[1][:futureSteps]

    error = mean_displacement_error([realX,realY],[predX,predY])
    return error

def nearestPD(A):
    B = (A + np.transpose(A)) / 2
    _, s, V = la.svd(B)
    H = np.dot(np.transpose(V), np.dot(np.diag(s), V))
    A2 = (B + H) / 2
    A3 = (A2 + np.transpose(A2)) / 2

    if is_positive_definite(A3):
        return A3

    spacing = np.spacing(la.norm(A))
    I = np.eye(A.shape[0])
    k = 1
    while not is_positive_definite(A3):
        mineig = np.min(np.real(la.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1

    return A3

#Returns true when input is positive-definite, via Cholesky
def is_positive_definite(B):
    try:
        _ = la.cholesky(B)
        return True
    except la.LinAlgError:
        return False
