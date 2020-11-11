"""
@author: karenlc
"""
from test_common import *
from gp_code.single_gp import singleGP
from utils.stats_trajectories import histogram

img         = mpimg.imread('imgs/goals.jpg')
station_img = mpimg.imread('imgs/train_station.jpg')
# Read the areas file, dataset, and form the goalsLearnedStructure object
goalsData, pathMat, __ = read_and_filter('parameters/CentralStation_GoalsDescriptions.csv','datasets/GCS/CentralStation_trainingSet.txt')

# We give the start and ending goals
startG = 0
nextG  = 6
histogram(pathMat[startG][nextG])
