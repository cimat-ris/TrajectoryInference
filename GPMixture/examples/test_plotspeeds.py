"""
@author: karenlc
"""
from test_common import *
from gp_code.single_gp import singleGP
from utils.stats_trajectories import trajectory_arclength, avg_speed, median_speed
import matplotlib.pyplot as plt
import random
from sklearn.linear_model import LinearRegression, HuberRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

imgGCS           = 'imgs/train_station.jpg'
# Read the areas file, dataset, and form the goalsLearnedStructure object
traj_dataset, goalsData, trajMat, __ = read_and_filter('GCS','parameters/CentralStation_GoalsDescriptions.csv','datasets/GC/Annotation/',use_pickled_data=True)


for i in range(goalsData.goals_n):
    for j in range(goalsData.goals_n):
        trajSet        = trajMat[i][j]
        # Only if we have enough trajectories
        if len(trajSet) < goalsData.min_traj_number:
            continue
        relativeSpeeds = []
        lengths        = []
        for tr in trajSet:
            # Times
            t = tr[2]
            # Average speed
            v = avg_speed(tr)
            # Arc lengths
            d = trajectory_arclength(tr)
            for k in range(1,len(t)):
                relativeSpeeds.append(float((d[k]-d[k-1])/(t[k]-t[k-1]))/v)
                lengths.append(d[k])
        lengths        = np.array(lengths).reshape(-1, 1)
        relativeSpeeds = np.array(relativeSpeeds)

        polyreg=make_pipeline(PolynomialFeatures(4),LinearRegression())
        polyreg.fit(lengths, relativeSpeeds)
        slenghts = np.sort(lengths, axis=0)
        relativeSpeedsPredicted = polyreg.predict(slenghts)
        p = plotter(title=f"Trajectories from {i} to {j}")
        p.set_background(imgGCS)
        p.plot_scene_structure(goalsData)
        p.plot_trajectories(trajMat[i][j])
        #p.show()
        fig,ax = plt.subplots(1)
        plt.plot(lengths,relativeSpeeds,'+')
        plt.plot(slenghts,relativeSpeedsPredicted)
        plt.ylabel('Relative speed')
        plt.xlabel('Length')
        plt.show()


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
    averageSpeeds.append(median_speed(tr))

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
