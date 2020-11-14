"""
@author: jbhayet
"""
from test_common import *
from utils.loader_ind import load_ind
import matplotlib.pyplot as plt

#img         = mpimg.imread('imgs/goals.jpg')
#station_img = mpimg.imread('imgs/train_station.jpg')
# Read the areas file, dataset, and form the goalsLearnedStructure object
#goalsData, pathMat, __ = read_and_filter('parameters/CentralStation_GoalsDescriptions.csv','datasets/GCS/CentralStation_trainingSet.txt')
#stepUnit  = 0.0438780780171   #get_number_of_steps_unit(pathMat, nGoals)

dataset_dir = "/home/jbhayet/opt/datasets/inD-dataset-v1.0/data/"
dataset_file= "00_tracks.csv"
traj_dataset= load_ind(dataset_dir+dataset_file)
traj_set    = traj_dataset.get_trajectories()
print(traj_set)
