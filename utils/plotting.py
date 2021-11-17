"""
Plotting functions
"""
import numpy as np
import pandas as pd
import random
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Ellipse
from matplotlib.animation import FuncAnimation
from utils.manip_trajectories import goal_center_and_size

color = ['lightgreen','springgreen','g','b','steelblue','y','tomato','orange','r','gold','yellow','lime',
         'cyan','teal','deepskyblue','dodgerblue','royalblue','blueviolet','indigo',
         'purple','magenta','deeppink','hotpink','sandybrown','darkorange','coral']


# Plot the goal
def plot_goals(ax,goal,color="green"):
    center, size = goal_center_and_size(goal)
    # Create rectangle patch
    rect = plt.Rectangle((center[0]-size[0]/2,center[1]-size[1]/2), size[0], size[1],facecolor=color, alpha=0.35)
    ax.add_patch(rect)

class multiple_plotter():
    def __init__(self,nplots=4,title=None):
        self.nplots        = nplots
        self.ncols         = 2
        self.nrows         = nplots//self.ncols
        self.figs,self.axs = plt.subplots(nrows=self.nrows,ncols=self.ncols)
        for i in range(self.nrows):
            for j in range(self.ncols):
                self.axs[i][j].margins(0, 0)
                self.axs[i][j].set_axis_off()
                self.axs[i][j].xaxis.set_major_locator(plt.NullLocator())
                self.axs[i][j].yaxis.set_major_locator(plt.NullLocator())
                if title!=None:
                    self.axs[i][j].set_title(title)
        plt.subplots_adjust(wspace=0.01, hspace=0.1)

    def set_background(self,img_background_path):
        # Use the image as a background
        img = mpimg.imread(img_background_path)
        s = img.shape
        v = [0,s[1],s[0],0]
        for i in range(self.nrows):
            for j in range(self.ncols):
                self.axs[i][j].imshow(img)
                self.axs[i][j].axis(v)

    # Plot the scene structure: goals and sub-goals
    def plot_scene_structure(self,goals_data, goals_likelihoods, draw_ids=False):
        inds = np.argsort(goals_likelihoods)
        inds = np.flip(inds)
        print(inds)
        for i in range(self.nrows):
            for j in range(self.ncols):
                idx = inds[i*self.ncols+j]
                for k in range(goals_data.goals_n):
                    center, size = goal_center_and_size(goals_data.goals_areas[k][1:])
                    if k!=idx:
                        plot_goals(self.axs[i][j],goals_data.goals_areas[k][1:])
                    else:
                        plot_goals(self.axs[i][j],goals_data.goals_areas[k][1:],color="blue")
                    if draw_ids:
                        self.axs[i][j].text(center[0], center[1],str(k),color='white')

    # Plot multiple predictions
    def plot_likeliest_predictions(self,observations, filtered_paths, predicted_paths, var_paths, goals_likelihoods,min_likelihood=0.001):
        # Plot the observed data
        for i in range(self.nrows):
            for j in range(self.ncols):
                self.axs[i][j].plot(observations[:,0],observations[:,1],'c')

        # Cycle over the potential goals
        inds = np.argsort(goals_likelihoods)
        inds = np.flip(inds)
        for idx in range(self.nplots):
            filt_path = filtered_paths[inds[idx]]
            pred_path = predicted_paths[inds[idx]]
            var_path  = var_paths[inds[idx]]
            likelihood= goals_likelihoods[inds[idx]]
            # For each goal, draws the prediction
            self.axs[idx//self.ncols][idx%self.ncols].plot(filt_path[:-1,0],filt_path[:-1,1],'b--')
            self.axs[idx//self.ncols][idx%self.ncols].plot(pred_path[:,0],pred_path[:,1],'b--')
            predictedN = pred_path.shape[0]
            # For the jth predicted element
            for j in range(predictedN):
                if j%2==0:
                    xy = [pred_path[j,0],pred_path[j,1]]
                    # It is: to have 3.0 sigmas. Then, the Ellipse constructor asks for the diameter, hence the 2.0
                    vx  = var_path[0,j,j]
                    vy  = var_path[1,j,j]
                    ell = Ellipse(xy,6.0*math.sqrt(math.fabs(vx)),6.0*math.sqrt(math.fabs(vy)))
                    ell.set_lw(1.0)
                    ell.set_fill(0)
                    ell.set_edgecolor('m')
                    self.axs[idx//self.ncols][idx%self.ncols].add_patch(ell)
            self.axs[idx//self.ncols][idx%self.ncols].set_title("Goal {}. Likelihood {:.2f}".format(inds[idx],likelihood))
    # Show the plots
    def show(self):
        plt.show()

class plotter():
    def __init__(self,title=None):
        self.fig,self.ax = plt.subplots(1)
        plt.margins(0, 0)
        plt.gca().set_axis_off()
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        self.ax.set_aspect('equal')
        if title!=None:
            self.ax.set_title(title)

    def clf(self):
        plt.clf()

    def set_background(self,img_background_path):
        # Use the image as a background
        img = mpimg.imread(img_background_path)
        self.ax.imshow(img)
        s = img.shape
        v = [0,s[1],s[0],0]
        plt.axis(v)

    # Plot the scene structure: goals and sub-goals
    def plot_scene_structure(self,goals_data, draw_ids=False):
        for i in range(goals_data.goals_n):
            plot_goals(self.ax,goals_data.goals_areas[i][1:])
            if draw_ids:
                center, size = goal_center_and_size(goals_data.goals_areas[i][1:])
                self.ax.text(center[0], center[1],str(i),color='white')

    # Plot the true data, the predicted ones and their variance
    def plot_prediction(self,observations, predicted, variances):
        observedX = observations[:,0]
        observedY = observations[:,1]
        self.ax.plot(observedX,observedY,'c',predicted[:,0],predicted[:,1],'b')
        self.ax.plot([observedX[-1],predicted[0,0]],[observedY[-1],predicted[0,1]],'b')
        predictedN = predicted.shape[0]
        for i in range(predictedN):
            xy = [predicted[i,0],predicted[i,1]]
            ell = Ellipse(xy,6.0*math.sqrt(math.fabs(variances[0][i,i])),6.0*math.sqrt(math.fabs(variances[1][i,i])))
            ell.set_lw(1.)
            ell.set_fill(0)
            ell.set_edgecolor('m')
            self.ax.add_patch(ell)

    # Plot the ground truth data
    def plot_ground_truth(self,gtPath):
        self.ax.plot(gtPath[:-1,0],gtPath[:-1,1],'c--')


    # Plot the filtered data
    def plot_filtered(self,filtered_paths):
        # Cycle over the potential goals
        for filtered_path in filtered_paths:
            if filtered_path.shape[0]>0:
                self.ax.plot(filtered_path[:-1,0],filtered_path[:-1,1],'b--')

    # Plot multiple predictions
    def plot_multiple_predictions_and_goal_likelihood(self,observations, paths, var_paths, goals_likelihood,min_likelihood=0.01):
        # Plot the observed data
        self.ax.plot(observations[:,0],observations[:,1],'c')

        maxLikelihood     = max(goals_likelihood)
        maxLikelihoodWidth= 2
        for i in range(len(goals_likelihood)):
            if (paths[i] is None) or goals_likelihood[i]<min_likelihood:
                continue
            # Line width function of the likelihood
            lw = (goals_likelihood[i]/maxLikelihood)*maxLikelihoodWidth
            # For each goal, draws the prediction
            self.ax.plot(paths[i][:,0],paths[i][:,1],'b--')
            self.ax.plot([observations[-1,0],paths[i][0,0]],[observations[-1,1],paths[i][0,1]],'b--')
            predictedN = paths[i].shape[0]
            # For the jth predicted element
            for j in range(predictedN):
                if j%2==0:
                    xy = [paths[i][j,0],paths[i][j,1]]
                    # It is: to have 3.0 sigmas. Then, the Ellipse constructor asks for the diameter, hence the 2.0
                    vx  = var_paths[i][0,j,j]
                    vy  = var_paths[i][1,j,j]
                    ell = Ellipse(xy,6.0*math.sqrt(math.fabs(vx)),6.0*math.sqrt(math.fabs(vy)))
                    ell.set_lw(lw)
                    ell.set_fill(0)
                    ell.set_edgecolor('m')
                    self.ax.add_patch(ell)

    # Plot a set of sample trajectories and an observed partial trajectory
    def plot_path_samples_with_observations(self,observations, paths, spec_color=None):
        samples = len(paths)
        if (samples == 0):
            return

        x = observations[:,0]
        y = observations[:,1]
        x.reshape((-1,1))
        y.reshape((-1,1))
        self.ax.plot(x,y,'c',lw=2.0)

        for i in range(samples):
            if paths[i][0] is not None:
                if spec_color is None:
                    randColor = random.choice(color)
                else:
                    randColor = spec_color
                samplex = paths[i][:,0]
                sampley = paths[i][:,1]
                samplex.reshape((-1,1))
                sampley.reshape((-1,1))

                self.ax.plot(samplex,sampley,color=randColor,alpha=0.5)

    # Plot a set of sample trajectories
    def plot_paths_samples_gt(self,paths_per_goals, n_samples=2):
        ngoals = len(paths_per_goals)
        for i in range(ngoals):
            for j in range(i):
                tr = paths_per_goals[i][j]
                n_samples_eff = min(n_samples,len(tr))
                idx= np.random.choice(len(tr), n_samples_eff, replace=False)
                for k in idx:
                    self.ax.plot(tr[k][:,0],tr[k][:,1],color="white",alpha=0.3)

    def plot_paths(self,traj_set, n_max=500):
        for i,tr in enumerate(traj_set):
            self.ax.plot(tr[0],tr[1])
            if i == n_max:
                break

    def pause(self,d):
        plt.pause(d)
        plt.cla()

    # Save the plots
    def save(self,file_name):
        plt.savefig(file_name,transparent=True,bbox_inches='tight')

    # Show the plots
    def show(self,block=True):
        plt.show(block=block)

    def clear(self):
        self.fig.clear()
#******************************************************************************#

# partial path and euclidian distance to goal
def plot_euclidean_distance_to_finish_point(img,trueX,trueY,knownN,finalXY):
    observedX = trueX[0:knownN]
    observedY = trueY[0:knownN]

    fig,ax = plt.subplots(1)
    ax.set_aspect('equal')
    ax.imshow(img) # Show the image

    plt.plot(observedX,observedY,'c', label='Partial path')
    lineX, lineY =[],[]
    lineX.append(trueX[knownN-1])
    lineX.append(finalXY[0])
    lineY.append(trueY[knownN-1])
    lineY.append(finalXY[1])
    plt.plot(lineX,lineY,'b', label='Euclidean distance')
    ax.legend()

    v = [0,img.shape[1],img.shape[0],0]
    plt.axis(v)
    plt.show()


def animate_multiple_predictions_and_goal_likelihood(img,x,y,nUsedData,nGoals,goals_likelihood,predictedXYVec,varXYVec,toFile):
    observedX = x[0:nUsedData]
    observedY = y[0:nUsedData]

    fig,ax = plt.subplots()
    plt.gca().set_axis_off()
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    # Plot the observed data
    lineObs, = ax.plot(observedX,observedY, lw=3,color='c')

    # Show the image
    ax.imshow(img)
    plt.tight_layout()
    # Likelihoods
    maxLikelihood = max(goals_likelihood)
    maxLW = 3.0

    # For all potential goals
    ells      = []
    pls       = []
    #ax.set_aspect('equal')
    for i in range(nGoals):
            lw = max((goals_likelihood[i]/maxLikelihood)*maxLW,1)
            e = Ellipse([0,0],0,0)
            e.set_fill(0)
            if (predictedXYVec[i].shape[0]==0):
                l, = ax.plot([0],[0], lw=0,color='b')
                e.set_lw(0)
            else:
                l, = ax.plot(predictedXYVec[i][0,0],predictedXYVec[i][0,1], lw=lw,color='b')
                ## Ellipse center
                xy  = [predictedXYVec[i][0,0],predictedXYVec[i][0,1]]
                vx  = varXYVec[i][0,0,0]
                vy  = varXYVec[i][1,0,0]
                e.center = xy
                e.width  = 6.0*math.sqrt(math.fabs(vx))
                e.height = 6.0*math.sqrt(math.fabs(vy))
                e.set_edgecolor('b')
                e.set_lw(lw)
            ax.add_patch(e)
            ells.append(e)
            pls.append(l)
    v = [0,img.shape[1],img.shape[0],0]
    plt.axis(v)


    def animate(i):
        for j in range(nGoals):
            if (predictedXYVec[j].shape[0]==0):
                continue
            predictedN = predictedXYVec[j].shape[0]
            if (i<predictedN):
                p   = [predictedXYVec[j][i,0],predictedXYVec[j][i,1]]
                vx  = varXYVec[j][0,i,i]
                vy  = varXYVec[j][1,i,i]
                pls[j].set_data(predictedXYVec[j][0:i,0],predictedXYVec[j][0:i,1])
                ells[j].center = p
                ells[j].width  = 6.0*math.sqrt(math.fabs(vx))
                ells[j].height = 6.0*math.sqrt(math.fabs(vy))


    anim = FuncAnimation(fig, animate,frames=100,interval=100)
    if toFile:
        anim.save('test-%d.mp4' % i)
    else:
        plt.show()
    return

# Plot a set of sample trajectories
def plot_path_samples(img,paths):
    n = len(paths)
    if(n == 0):
        return
    fig,ax = plt.subplots(1)
    ax.set_aspect('equal')
    # Show the image
    ax.imshow(img)
    for p in paths:
        plt.plot(p[:,0],p[:,1])
    s = img.shape
    v = [0,s[1],s[0],0]
    plt.axis(v)
    plt.show()


def plot_observations_predictive_mean_and_sample(img,traj,knownN,predXY,sampleXY):
    x, y = traj[0], traj[1]
    observedX = x[:knownN]
    observedY = y[:knownN]

    fig,ax = plt.subplots(1)
    ax.set_aspect('equal')
    ax.imshow(img)

    plt.plot(observedX,observedY,'b',lw=2.0)        #observed trajectory
    plt.plot(x[knownN:],y[knownN:],'b--',lw=2.0)    #true trajectory
    plt.plot(sampleXY[0],sampleXY[1],'g--',lw=2.0)  #sample
    plt.plot(predXY[0],predXY[1],'c--',lw=2.0)      #predictive mean
    s = img.shape
    v = [0,s[1],s[0],0]
    plt.axis(v)
    plt.show()

def sequence_of_observations_predmean_samples(img,traj,knownN,predXY,sampleXY):
    #plot a matrix of images with different sample and prediction
    N        = len(knownN) # Number of images
    n, m = 1, 1
    if(N%3 == 0):
        n = 3
        m = int(N/3)
    elif(N%2 == 0):
        n = 2
        m = int(N/2)
    else:
        m = N
    fig, axes = plt.subplots(n,m)
    x, y = traj[0], traj[1]

    for i in range(n):
        for j in range(m):
            axes[i,j].set_aspect('equal')
            # Show the image
            axes[i,j].imshow(img)
            # Plot each test
            k = (i*m)+j
            observedX = x[:knownN[k]+1]
            observedY = y[:knownN[k]+1]

            axes[i,j].plot(observedX,observedY,'b',lw=2.0)              #observed trajectory
            axes[i,j].plot(x[knownN[k]:],y[:knownN[k]+1],'b--',lw=2.0)  #true trajectory
            axes[i,j].plot(predXY[k][0],predXY[k][1],'g--',lw=2.0)      #predictive mean
            axes[i,j].plot(sampleXY[k][0],sampleXY[k][1],'c--',lw=2.0)  #sample
            axes[i,j].axis('off')
    plt.show()


#Grafica con subplots los tests del sampling dado un conjunto de observaciones
def plot_interaction_with_sampling_test(img,observedTraj, samplesVec, potentialVec):
    N        = len(samplesVec) # Number of joint samples
    numPaths = len(observedTraj)
    n, m = 1, 1
    if(N%3 == 0):
        n = 3
        m = int(N/3)
    elif(N%2 == 0):
        n = 2
        m = int(N/2)
    else:
        m = N
    fig, axes = plt.subplots(n,m)
    for i in range(n):
        for j in range(m):
            axes[i,j].set_aspect('equal')
            # Show the image
            axes[i,j].imshow(img)
            # Plot each test
            t = (i*m)+j
            for k in range(numPaths):
                col = (k)%len(color)
                observedX, observedY = observedTraj[k][0],observedTraj[k][1]
                axes[i,j].plot(observedX,observedY,color[col],lw=2.0)

                sampleX, sampleY = samplesVec[t][k][0],samplesVec[t][k][1]
                axes[i,j].plot(sampleX,sampleY,color[col]+'--',lw=2.0)
                string = "{0:1.3e}".format(potentialVec[t])
                axes[i,j].set_title('w = '+string)
            axes[i,j].axis('off')
    plt.show()

#Grafica con subplots los tests del sampling dado un conjunto de observaciones
def plot_interaction_test_weight_and_error(img,observedPaths, samplesVec, potentialVec, errorVec):
    N        = len(samplesVec) # Number of joint samples
    numPaths = len(observedPaths)
    n, m = 1, 1
    if(N%3 == 0):
        n = 3
        m = int(N/3)
    elif(N%2 == 0):
        n = 2
        m = int(N/2)
    else:
        m = N
    fig, axes = plt.subplots(n,m)
    for i in range(n):
        for j in range(m):
            axes[i,j].set_aspect('equal')
            # Show the image
            axes[i,j].imshow(img)
            # Plot each test
            t = (i*m)+j
            for k in range(numPaths):
                colorId = (k)%len(color)
                obsx,oy = observedPaths[k].x,observedPaths[k].y
                axes[i,j].plot(obsx,oy,color[colorId],lw=2.0)
                sx,sy = samplesVec[t][k].x,samplesVec[t][k].y
                axes[i,j].plot(sx,sy,color[colorId]+'--',lw=2.0)
                w = "{0:1.3e}".format(potentialVec[t])
                error = "{0:.2f}".format(errorVec[t])
                axes[i,j].set_title('w = '+w+'\nerror = '+error)
            axes[i,j].axis('off')
    plt.show()

def plot_interaction_with_sampling(img, observedPaths, samples, potential, error):
    n = len(observedPaths)

    fig, ax = plt.subplots(1)
    ax.set_aspect('equal')
    ax.imshow(img)
    for i in range(n):
        ind = i%len(color)
        Color = color[ind]
        plt.plot(observedPaths[i].x, observedPaths[i].y,Color,lw=2.0)
        plt.plot(samples[i].x, samples[i].y,Color+'--',lw=2.0)
        strPotential = "{0:1.3e}".format(potential)
        strError = "{0:.2f}".format(error)
    plt.title('w = '+strPotential + '\nerror = '+strError)
    #plt.title('\nerror = '+strError)
    s = img.shape
    v = [0,s[1],s[0],0]
    plt.axis(v)
    plt.show()

def plot_table(data, rowLabels, colLabels, tittle):
    stringData = []
    n, m = len(data), len(data[0])
    for i in range(n):
        row = []
        for j in range(m):
            string = "{0:1.3e}".format(data[i][j])
            row.append(string)
        stringData.append(row)
    fig, ax = plt.subplots()
    plt.title(tittle)
    # hide axes
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')

    ax.table(cellText=stringData,cellLoc='center',rowLabels = rowLabels, colLabels = colLabels, loc='center')

    fig.tight_layout()

    plt.show()

def boxplot(data, title):
    boxprops = dict(linewidth=1.2, color='black')
    flierprops = dict(marker='o', markerfacecolor='blue', markersize=8)
    meanlineprops = dict(linestyle='--', linewidth=2.3, color='green')

    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.boxplot(data, boxprops=boxprops, whiskerprops=dict(linestyle='-', linewidth=1.2, color='black'), flierprops=flierprops, meanprops=meanlineprops, showmeans=True, meanline=True)

#plot an array of data in multiple boxplots
def multiple_boxplots(data, labels):
    num_boxes = len(data)
    blackbox = dict(linewidth=1.2, color='black')
    bluebox = dict(linewidth=1.2, color='blue')
    flierprops = dict(marker='o', markerfacecolor='blue', markersize=8)
    meanlineprops = dict(linestyle='--', linewidth=2.3, color='green')

    boxprops = []
    for i in range(num_boxes):
        if i%2 == 0:
            boxprops.append(bluebox)
        else:
            boxprops.append(blackbox)


    fig, ax = plt.subplots()
    ax.boxplot(data, labels = labels, boxprops=blackbox, whiskerprops=dict(linestyle='-', linewidth=1.2, color='black'))
    plt.show()

def joint_multiple_boxplots(data_a, data_b, title):
    color_a = 'm'
    color_b = '#2C7BB6'
    ticks = ['1/5', '2/5', '3/5', '4/5']

    plt.figure()

    bp_a = plt.boxplot(data_a, positions=np.array(range(len(data_a)))*2.0-0.4, widths=0.6, whiskerprops=dict(linestyle='-', linewidth=1.2, color=color_a), showfliers=True)
    bp_b = plt.boxplot(data_b, positions=np.array(range(len(data_b)))*2.0+0.4, widths=0.6, whiskerprops=dict(linestyle='-', linewidth=1.2, color=color_b), showfliers=True)
    set_box_color(bp_a, color_a)
    plt.setp(bp_a['medians'], color='indigo')
    set_box_color(bp_b, color_b)
    plt.setp(bp_b['medians'], color='blue')

    plt.title(title)
    #legend
    plt.plot([], c=color_a, label='Predictive mean')
    plt.plot([], c=color_b, label='Best of samples')
    plt.legend()

    plt.xticks(range(0, len(ticks) * 2, 2), ticks)
    plt.xlim(-2, len(ticks)*2)
    maxy = 0
    for i in range(len(data_a)):
        current_max = max(data_a[i])
        if current_max > maxy:
            maxy = current_max
    plt.ylim(0, maxy)

    plt.xlabel('Observed data')
    plt.ylabel('Error in pixels')
    plt.tight_layout()

def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
