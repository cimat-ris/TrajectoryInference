"""
Created on Fri Mar 19 19:46:05 2021

@author: karen
"""
import random
from test_common import *
from gp_code.kernels import *
from gp_code.mixture_gpT import mixtureGPT
from utils.manip_trajectories import start_time

# Read the areas file, dataset, and form the goalsLearnedStructure object
goalsDescriptions= '../parameters/CentralStation_GoalsDescriptions.csv'
trajFile         = '../datasets/GC/Annotation/'
imgGCS           = '../imgs/train_station.jpg'

traj_dataset, goalsData, trajMat, filtered = read_and_filter('GCS',goalsDescriptions,trajFile,use_pickled_data=True)

goalsData.kernelsX = create_kernel_matrix('combinedTrautman', goalsData.nGoals, goalsData.nGoals)
goalsData.kernelsY = create_kernel_matrix('combinedTrautman', goalsData.nGoals, goalsData.nGoals)

"""**********          Testing          ***********"""

# Sort by start time
filtered.sort(key=start_time)