"""
@author: karenlc
"""
from test_common import *
from gp_code.single_gp import singleGP
from utils.manip_trajectories import goal_centroid

# Read the areas file, dataset, and form the goalsLearnedStructure object
goalsDescriptions= './parameters/CentralStation_GoalsDescriptions.csv'
trajFile         = './datasets/GC/Annotation/'
imgGCS           = './imgs/train_station.jpg'
img              = mpimg.imread(imgGCS)

traj_dataset, goalsData, trajMat, __ = read_and_filter('GCS',goalsDescriptions,trajFile,use_pickled_data=True)

# Selection of the kernel type
kernelType = "linePriorCombined"
nParameters = 4

# Read the kernel parameters from file
goalsData.kernelsX = read_and_set_parameters("parameters/linearpriorcombined20x20_x.txt",nParameters)
goalsData.kernelsY = read_and_set_parameters("parameters/linearpriorcombined20x20_y.txt",nParameters)

# Sampling 3 trajectories between all the pairs of goals
vecX, vecY = [], []
for i in range(goalsData.nGoals):
    for j in range(i,goalsData.nGoals):
        if(i != j):
            iCenter = goal_centroid(goalsData.areas_coordinates[i])
            jCenter = goal_centroid(goalsData.areas_coordinates[j])
            # The basic element here is this object, that will do the regression work
            gp = singleGP(i,j,goalsData)
            likelihood = gp.update([iCenter[0]],[iCenter[1]],[0.0])
            predictedXY,varXY = gp.predict()
            # Generate samples
            nvecX,nvecY       = gp.generate_samples(1)
            vecX = vecX + nvecX
            vecY = vecY + nvecY

plot_path_samples(img, vecX,vecY)
