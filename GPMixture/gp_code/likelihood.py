"""
Error and likelihood evaluation
"""
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import numpy as np
import math
from numpy import linalg as la
from utils.stats_trajectories import euclidean_distance
from utils.manip_trajectories import goal_center_and_size

# Mean euclidean distance between true and predicted data
def mean_euc_error(true, predicted):
    u = np.array(true)
    v = np.array(predicted)

    return la.norm(u-v)

# Mean absolute error (mx,my) between true and predicted data
def mean_abs_error(trueX, trueY, predX, predY):
    trueLen, predLen = len(trueX), len(predX)
    mx = mean_absolute_error(trueX[trueLen -predLen:-1], predX)
    my = mean_absolute_error(trueY[trueLen -predLen:-1], predY)

    return [mx, my]

#average_displacement_error
# Average L2 distance between ground truth and our prediction
def ADE(true_xy, prediction_xy):
    minLen = min( len(true_xy[0]),len(prediction_xy[0]) )

    true_x = np.array(true_xy[0][:minLen])
    true_y = np.array(true_xy[1][:minLen])
    pred_x = np.array(prediction_xy[0][:minLen])
    pred_y = np.array(prediction_xy[1][:minLen])

    r = np.sqrt( (true_x - pred_x)**2 + (true_y - pred_y)**2 )
    return np.mean(r)

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

# Compute ADE and FDE
def ADE_FDE(full_path, predicted_xy, observed, future_steps):
    real_x = full_path[0][observed : observed+future_steps]
    real_y = full_path[1][observed : observed+future_steps]
    pred_x = predicted_xy[0][:future_steps]
    pred_y = predicted_xy[1][:future_steps]

    real_x_l = full_path[0][observed+future_steps-1:observed+future_steps]
    real_y_l = full_path[1][observed+future_steps-1:observed+future_steps]
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
