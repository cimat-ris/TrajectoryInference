"""
@author: karenlc
"""
from test_common import *
from gp_code.single_gp import singleGP
from utils.manip_trajectories import middle_of_area

img         = mpimg.imread('imgs/goals.jpg')
station_img = mpimg.imread('imgs/train_station.jpg')
# Read the areas file, dataset, and form the goalsLearnedStructure object
goalsData, pathMat, __ = read_and_filter('parameters/CentralStation_areasDescriptions.csv','datasets/CentralStation_trainingSet.txt')
# TODO: we should remove this parameter; a priori it could be deduced in some way with the speed
stepUnit  = 0.0438780780171   #get_number_of_steps_unit(pathMat, nGoals)

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

# Sampling 3 trajectories between all the pairs of goals
vecX, vecY = [], []
for i in range(goalsData.nGoals):
    for j in range(i,goalsData.nGoals):
        if(i != j):
            iCenter = middle_of_area(goalsData.areas[i])
            jCenter = middle_of_area(goalsData.areas[j])
            # The basic element here is this object, that will do the regression work
            gp = singleGP(i,j,stepUnit,goalsData)
            likelihood = gp.update([iCenter[0]],[iCenter[1]],[0.0])
            predictedXY,varXY = gp.predict()
            # Generate samples
            nvecX,nvecY       = gp.generate_samples(10)
            vecX = vecX + nvecX
            vecY = vecY + nvecY

plot_path_samples(img, vecX,vecY)
