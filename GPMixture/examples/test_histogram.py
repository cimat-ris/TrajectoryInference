"""
@author: karenlc
"""
from test_common import *
from gp_code.single_gp import singleGP
from utils.stats_trajectories import tr_histogram

# Read the areas file, dataset, and form the goalsLearnedStructure object
coordinates      = "img"
traj_dataset, goalsData, trajMat, __ = read_and_filter('GCS',coordinate_system=coordinates,use_pickled_data=True)


# We give the start and ending goals
startG = 0
nextG  = 6
tr_histogram(trajMat[startG][nextG])
