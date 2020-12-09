"""
Error and likelihood evaluation
"""
import numpy as np
import math
from numpy import linalg as la
from gp_code.regression import prediction_xy
from utils.io_misc import euclidean_distance
from utils.manip_trajectories import goal_center_and_size
from utils.manip_trajectories import goal_centroid

D = 150. #value for compute_goal_likelihood

# Mean euclidean distance between true and predicted data
def mean_euc_error(u,v):
    error = 0.
    for i in range(len(u)):
        error += math.sqrt((u[i]- v[i])**2)
    return error/len(u)

# Mean absolute error (mx,my) between true and predicted data
def mean_abs_error(trueX, trueY, predX, predY):
    lp, l = len(predX), len(trueX)
    ex, ey = 0.0, 0.0
    for i in range(lp):
        ex += abs(trueX[l-1-i]-predX[lp-1-i])
        ey += abs(trueY[l-1-i]-predY[lp-1-i])
    mx = ex/len(predX)
    my = ey/len(predY)
    return [mx, my]

#average_displacement_error
# Average L2 distance between ground truth and our prediction
def ADE(true_XY, prediction_XY):
    error = 0.
    trueX, trueY = true_XY[0], true_XY[1]
    predictionX, predictionY = prediction_XY[0], prediction_XY[1]
    l = min(len(trueX),len(predictionX))
    for i in range(l):
        error += math.sqrt((trueX[i]-predictionX[i])**2 + (trueY[i]-predictionY[i])**2)
    if(l>0):
        error = error/l
    else:
        return -1.0
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

# For a given dataset (knownX,knownY,knownL),
# takes half of the data as known and predicts the remaining half. Then, evaluate the likelihood.
def likelihood_from_partial_path(nPoints,observedX,observedY,observedL,startG,finishG,goalsData):
    error = compute_prediction_error_of_points_along_the_path(nPoints,observedX,observedY,observedL,startG,finishG,goalsData)
    return (math.exp(-1.*( error**2)/D**2 ))

# For a given dataset (knownX,knownY,knownL), takes half of the data as known
# and predicts the remaining half. Then, evaluate the prediction error.
def compute_prediction_error_of_points_along_the_path(nPoints,observedX,observedY,observedL,startG,finishG,goalsData):
    # Known data
    observedN = len(observedX)
    halfN     = max(1,int(observedN/2))

    # First half of the known data
    trueX = observedX[0:halfN]
    trueY = observedY[0:halfN]
    trueL = observedL[0:halfN]
    # Get the last point and add it to the observed data
    finishXY,__ = goal_center_and_size(goalsData.areas_coordinates[finishG])
    finishD     = euclidean_distance([trueX[len(trueX)-1],trueY[len(trueY)-1]],finishXY)
    np.append(trueX,finishXY[0])
    np.append(trueY,finishXY[1])
    np.append(trueL,finishD*goalsData.units[startG][finishG])

    d = int(halfN/nPoints)
    if d<1:
        return 1.0
    # Prepare the ground truths and the list of l to evaluate
    realX         = observedX[halfN:halfN+nPoints*d:d]
    realY         = observedY[halfN:halfN+nPoints*d:d]
    predictionSet = observedL[halfN:halfN+nPoints*d:d]
    # Get the prediction based on the GP
    predX, predY, varX,varY = prediction_xy(trueX,trueY,trueL, predictionSet, goalsData.kernelsX[startG][finishG],goalsData.kernelsY[startG][finishG])
    # Evaluate the error
    return ADE([realX,realY],[predX,predY])

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

# Compute ADE and FDE
def ADE_FDE(full_path, predicted_xy, observed, future_steps):
    real_x = full_path.x[observed : observed+future_steps]
    real_y = full_path.y[observed : observed+future_steps]
    pred_x = predicted_xy[0][:future_steps]
    pred_y = predicted_xy[1][:future_steps]

    real_x_l = full_path.x[observed+future_steps-1:observed+future_steps]
    real_y_l = full_path.y[observed+future_steps-1:observed+future_steps]
    pred_x_l = predicted_xy[0][:future_steps]
    pred_y_l = predicted_xy[1][:future_steps]

    #***Both ADE?
    return ADE([real_x,real_y],[pred_x,pred_y]), ADE([real_x_l,real_y_l],[pred_x_l,pred_y_l])

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
