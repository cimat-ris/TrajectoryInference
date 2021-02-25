"""
Test GP misture | Trautman
"""
import random
from test_common import *
from gp_code.single_gp import singleGP
from gp_code.kernels import *

# Read the areas file, dataset, and form the goalsLearnedStructure object
goalsDescriptions= '../parameters/CentralStation_GoalsDescriptions.csv'
trajFile         = '../datasets/GC/Annotation/'
imgGCS           = '../imgs/train_station.jpg'

traj_dataset, goalsData, trajMat, __ = read_and_filter('GCS',goalsDescriptions,trajFile,use_pickled_data=True)

#print('--- goal transition ---')
#print(goalsData.priorTransitions)

for i in range(goalsData.nGoals):
    s = np.sum(goalsData.priorTransitions[i])
    if s != 1.0:
        print('--- goal ',i,' ---')
        print(goalsData.priorTransitions[i])
        print('sum = ', s)
    