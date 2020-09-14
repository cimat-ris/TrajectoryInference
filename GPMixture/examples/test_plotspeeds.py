"""
@author: karenlc
"""
from test_common import *
from gp_code.single_gp import singleGP
from utils.stats_trajectories import get_data_from_paths
import matplotlib.pyplot as plt

img         = mpimg.imread('imgs/goals.jpg')
station_img = mpimg.imread('imgs/train_station.jpg')
# Read the areas file, dataset, and form the goalsLearnedStructure object
goalsData, pathMat, __ = read_and_filter('parameters/CentralStation_GoalsDescriptions.csv','datasets/CentralStation_trainingSet.txt')
stepUnit  = 0.0438780780171   #get_number_of_steps_unit(pathMat, nGoals)

# We give the start and ending goals
startG  = 0
nextG   = 6
x,y,t,l,s = get_data_from_paths(pathMat[startG][nextG])

fig,ax = plt.subplots(1)
n = min(len(s),30)
for i in range(n):
    plt.plot(s[i])
plt.ylabel('Speed')
plt.xlabel('Time step')
plt.show()
