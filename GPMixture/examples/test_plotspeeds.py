"""
@author: karenlc
"""
from test_common import *
from gp_code.single_gp import singleGP
from utils.stats_trajectories import trajectory_arclength
import matplotlib.pyplot as plt
import random

# Read the areas file, dataset, and form the goalsLearnedStructure object
traj_dataset, goalsData, trajMat, __ = read_and_filter('GCS','../parameters/CentralStation_GoalsDescriptions.csv','../datasets/GC/Annotation/',use_pickled_data=True)

trajSet = trajMat[0][6]
localSpeed = []
for tr in trajSet:
    t = tr[2]
    d = trajectory_arclength(tr)
    trSpeed = [0.0]
    for i in range(1,len(t)):
        trSpeed.append(float( (d[i]-d[i-1])/(t[i]-t[i-1]) ) )
    localSpeed.append(trSpeed)


fig,ax = plt.subplots(1)

""" #This trajectory has an abnormal value at the end
traj = trajSet[130]
for i in range(len(traj[0])):
    plt.plot(traj[0][i],traj[1][i], 'b+' )
"""
#Plot a random sample of the trajectory set
sample = random.choices(range(len(trajSet)), k=30)

for i in range(len(sample)):
    plt.plot(localSpeed[sample[i]])
plt.ylabel('Speed')
plt.xlabel('Time')
plt.show()
