"""
@author: karenlc
"""
from test_common import *
from gp_code.single_gp import singleGP
from utils.manip_trajectories import goal_centroid

# Read the areas file, dataset, and form the goalsLearnedStructure object
trajFile         = 'datasets/GC/'
imgGCS           = 'imgs/train_station.jpg'
img              = mpimg.imread(imgGCS)
coordinates      = "img"
traj_dataset, goalsData, trajMat, __, __ = read_and_filter('GCS','datasets/GC/',use_pickled_data=True,coordinate_system=coordinates)
# Selection of the kernel type
kernelType = "linePriorCombined"
nParameters = 4

# Read the kernel parameters from file
goalsData.kernelsX = read_and_set_parameters("parameters/linearpriorcombined20x20_x.txt",nParameters)
goalsData.kernelsY = read_and_set_parameters("parameters/linearpriorcombined20x20_y.txt",nParameters)

# Sampling 3 trajectories between all the pairs of goals
allPaths = []
for i in range(goalsData.goals_n):
    for j in range(i,goalsData.goals_n):
        if(i != j) and len(trajMat[i][j])>0 and goalsData.kernelsX[i][j].optimized is True:
            path  = trajMat[i][j][0]
            pathX, pathY, pathT = path
            pathL = trajectory_arclength(path)
            observations, __ = observed_data([pathX,pathY,pathL,pathT],2)
            if observations is None:
                continue
            # The basic element here is this object, that will do the regression work
            gp = singleGP(i,j,goalsData)
            likelihood = gp.update(observations)
            # Generate samples
            predictedXY,varXY = gp.predict_path(compute_sqRoot=True)
            paths             = gp.sample_paths(3)
            allPaths          = allPaths + paths

plot_path_samples(img,allPaths)
