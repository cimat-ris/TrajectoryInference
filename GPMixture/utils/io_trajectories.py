import numpy as np
from gp_code.trajectory import trajectory
from gp_code.goal_pairs import goal_pairs
from utils.loaders.loader_ind import load_ind
from utils.loaders.loader_gcs import load_gcs
from utils.loaders.loader_edinburgh import load_edinburgh
from utils.manip_trajectories import *
import pandas as pd

""" Alternative functions, without the class trajectory """
#new get_paths_from_file_new
def get_traj_from_file(dataset_id,dataset_traj,goals,coordinate_system='img'):
    traj_dataset= load_gcs(dataset_traj, coordinate_system=coordinate_system)
    traj_set    = traj_dataset.get_trajectories()
    print("[INF] Loaded {:s} set, length: {:03d} ".format(dataset_id,len(traj_set)))

    trajectories = []
    for tr in traj_set:
        x, y, t = tr[:,0], tr[:,1], tr[:,4]
        #NOTE: Should we check if there are repeat positions in the traj?
        trajectories.append([x,y,t])
        
    multigoal_traj = multigoal_trajectories(trajectories, goals)
    #*** break multigoal traj + add them to the traj set
    return trajectories
    

"""-----------------------------------------------------"""

def get_paths_from_file_new(dataset_id,dataset_path,areas,coordinate_system='img'):
    paths, multigoal_paths = [],[]
    traj_dataset= load_gcs(dataset_path,coordinate_system=coordinate_system)
    traj_set    = traj_dataset.get_trajectories()
    print("[INF] Loaded {:s} set, length: {:03d} ".format(dataset_id,len(traj_set)))

    # Each line should contain a path
    for t in traj_set:
        auxT = t[:,4]
        auxX = t[:,0]
        auxY = t[:,1]
        auxPath = trajectory(auxT,auxX,auxY)
        gi = get_goal_sequence(auxPath,areas)
        if len(gi) > 2:
            multigoal_paths.append(auxPath)
            new_paths = break_multigoal_path(auxPath,gi,areas)
            for j in range(len(new_paths)):
                paths.append(new_paths[j])
        else:
                paths.append(auxPath)
    return traj_dataset, paths, multigoal_paths

def get_uncut_paths_from_file(file):
    paths = []
    # Open file
    with open(file) as f:
        # Each line should contain a path
        for line in f:
            dataX, dataY, dataT = [],[],[]
            # Split the line into sub-strings
            data = line.split()
            for i in range(0, len(data), 3):
                x_ = int(data[i])
                y_ = int(data[i+1])
                t_ = int(data[i+2])
                if equal(dataX,dataY,x_,y_) == 0: #checks if the last point of data is the same as (x_,y_), if it is then its discarded
                    dataX.append(x_)
                    dataY.append(y_)
                    dataT.append(t_)
            newPath = trajectory(dataT,dataX,dataY)
            paths.append(newPath)
    return paths

"""********** READ AND FILTER DATA and DEFINE THE STRUCTURES **********"""
def read_and_filter_new(dataset_id,areas_file,trajectories_file):
    # Read the areas data from the specified file
    data     = pd.read_csv(areas_file)
    areas    = data.values[:,2:]
    areasAxis= data.values[:,1]
    nGoals   = len(areas)
    # We process here multi-objective trajectories into sub-trajectories
    traj_dataset,dataPaths,__ = get_paths_from_file_new(dataset_id,trajectories_file,areas)
        #usefulPaths  = get_paths_in_areas(dataPaths,areas)
        #trajMat, arclenMat = define_trajectories_start_and_end_areas(areas,areas,usefulPaths)
    # Get useful paths and split the trajectories into pairs of goals
    trajMat, arclenMat = separate_trajectories_between_goals(dataPaths, areas)
    
    # Remove the trajectories that are either too short or too long
    pathMat, learnSet = filter_path_matrix(trajMat, nGoals, nGoals)
    sortedPaths = sorted(learnSet, key=time_compare)
    print("[INF] Number of filtered paths: ",len(learnSet))
    # Form the object goalsLearnedStructure
    goalsData = goal_pairs(areas,areasAxis,pathMat)
    return traj_dataset,goalsData,pathMat,learnSet

def read_and_filter(areasFile,trajectoriesFile):
    # Read the areas data from a file
    data     = pd.read_csv(areasFile)
    # TODO: handle the 6
    areas    = data.values[:,2:]
    areasAxis= data.values[:,1]
    nGoals   = len(areas)
    # We process here multi-objective trajectories into sub-trajectories
    dataPaths,__ = get_paths_from_file('datasets/GCS/CentralStation_trainingSet.txt',areas)
    usefulPaths  = get_paths_in_areas(dataPaths,areas)
    print("[INF] Number of useful paths: ",len(usefulPaths),"/",len(dataPaths))
    # Split the trajectories into pairs of goals
    trajMat, arclenMat = define_trajectories_start_and_end_areas(areas,areas,usefulPaths)
    # Remove the trajectories that are either too short or too long
    pathMat, learnSet = filter_path_matrix(trajMat, nGoals, nGoals)
    sortedPaths = sorted(learnSet, key=time_compare)
    print("[INF] Number of filtered paths: ",len(learnSet))
    # Form the object goalsLearnedStructure
    goalsData = goal_pairs(areas,areasAxis,pathMat)
    return goalsData,pathMat,learnSet
