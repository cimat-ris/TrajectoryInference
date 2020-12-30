"""
@author: karenlc
"""
from test_common import *
from gp_code.single_gp import singleGP
from utils.stats_trajectories import trajectory_arclength, avg_speed
import matplotlib.pyplot as plt
import random

# Read the areas file, dataset, and form the goalsLearnedStructure object
traj_dataset, goalsData, trajMat, __ = read_and_filter('GCS','../parameters/CentralStation_GoalsDescriptions.csv','../datasets/GC/Annotation/',use_pickled_data=True)

trajSet       = trajMat[0][6]
instantSpeeds = []
averageSpeeds = []
# For all the trajectories
for tr in trajSet:
    # Times
    t = tr[2]
    # Arc lengths
    d = trajectory_arclength(tr)
    trajectorySpeeds = [0.0]
    for i in range(1,len(t)):
        trajectorySpeeds.append(float( (d[i]-d[i-1])/(t[i]-t[i-1]) ) )
    instantSpeeds.append(trajectorySpeeds)
    averageSpeeds.append(avg_speed(tr))


""" #This trajectory has an abnormal value at the end
traj = trajSet[130]
for i in range(len(traj[0])):
    plt.plot(traj[0][i],traj[1][i], 'b+' )

"""
# Plot a random sample of the trajectory set
fig,ax = plt.subplots(1)
sample = random.choices(range(len(trajSet)), k=30)
for i in range(len(sample)):
    plt.plot(instantSpeeds[sample[i]])
plt.ylabel('instantaneous speed')
plt.xlabel('Time')
plt.show()

fig,ax = plt.subplots(1)
sample = random.choices(range(len(trajSet)), k=30)
for i in range(len(sample)):
    plt.plot(instantSpeeds[sample[i]]/averageSpeeds[sample[i]])
plt.ylabel('Relative speed')
plt.xlabel('Time')
plt.show()
