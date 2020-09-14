import numpy as np
import math
from gp_code.trajectory import trajectory
from gp_code.goal_pairs import goal_pairs
from utils.manip_trajectories import *
import pandas as pd

"""********** READ DATA **********"""
# Lectura de los nombres de los archivos de datos
def readDataset(fileName):
    file = open(fileName,'r')
    lines = file.readlines()
    for i in range(len(lines)):
        lines[i] = lines[i].strip("\n")
    return lines


def get_paths_from_file(path_file,areas):
    paths, multigoal_paths = [],[]
    # Open file
    with open(path_file) as f:
        # Each line should contain a path
        for line in f:
            auxX, auxY, auxT = [],[],[]
            # Split the line into sub-strings
            data = line.split()
            for i in range(0, len(data), 3):
                x_ = int(data[i])
                y_ = int(data[i+1])
                t_ = int(data[i+2])
                if equal(auxX,auxY,x_,y_) == 0:
                    auxX.append(x_)
                    auxY.append(y_)
                    auxT.append(t_)
            auxPath = trajectory(auxT,auxX,auxY)
            gi = get_goal_sequence(auxPath,areas)
            if len(gi) > 2:
                multigoal_paths.append(auxPath)
                new_paths = break_multigoal_path(auxPath,gi,areas)
                for j in range(len(new_paths)):
                    paths.append(new_paths[j])
            else:
                paths.append(auxPath)
    return paths, multigoal_paths

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


# Reads the set of files and gets the points (x,y,z),
# z = {time, arc-len} according to flag = {"time", "length"}
def get_data_from_files(files, flag):
    for i in range(len(files)):
        auxX, auxY, auxT = read_file(files[i])
        auxL = arclength(auxX, auxY)
        if(i==0):
            x, y, t = [auxX], [auxY], [auxT]
            l = [auxL]
        else:
            x.append(auxX)
            y.append(auxY)
            t.append(auxT)
            l.append(auxL)

    if(flag == "time"):
        z = t
    if(flag == "length"):
        z = l
    return x, y, z

# Writes in a file the paths that we can use for the algorithm
def write_useful_paths_file(paths): #paths es un vector de indices
    N = len(paths)
    #f = open("usefulPaths_%d.txt"%N,"w")
    f = open("usefulPaths.txt","w")
    for j in range(N):
        i = paths[j]
        if i < 10:
            s = "Data/00000%d.txt\n"%(i)
            f.write(s)
        if i >= 10 and i < 100:
            s = "Data/0000%d.txt\n"%(i)
            f.write(s)
        if i >= 100 and i < 1000:
            s = "Data/000%d.txt\n"%(i)
            f.write(s)
        if i >= 1000 and i <= 2000:
            s = "Data/00%d.txt\n"%(i)
            f.write(s)
    f.close()


def write_data(data, fileName):
    n = len(data)
    f = open(fileName,"w+")
    for i in range(n):
        s = "%d\n"%(data[i])
        f.write(s)
    f.close()

def read_data(fileName):
    data = []
    f = open(fileName,'r')
    lines = f.readlines()
    for i in range(len(lines)):
        lines[i] = lines[i].strip("\n")
        data.append( float(lines[i]) )
    f.close()
    return data

"""********** READ AND FILTER DATA and DEFINE THE STRUCTURES **********"""
def read_and_filter(areasFile,traectoriesFile):
    # Read the areas data from a file
    data     = pd.read_csv(areasFile)
    # TODO: handle the 6
    areas    = data.values[:,2:]
    areasAxis= data.values[:,1]
    nGoals   = len(areas)
    # We process here multi-objective trajectories into sub-trajectories
    dataPaths,__ = get_paths_from_file('datasets/CentralStation_trainingSet.txt',areas)
    usefulPaths          = get_paths_in_areas(dataPaths,areas)
    print("[INF] Number of useful paths: ",len(usefulPaths),"/",len(dataPaths))
    # Split the trajectories into pairs of goals
    startToGoalPath, arclenMat = define_trajectories_start_and_end_areas(areas,areas,usefulPaths)
    # Remove the trajectories that are either too short or too long
    pathMat, learnSet = filter_path_matrix(startToGoalPath, nGoals, nGoals)
    sortedPaths = sorted(learnSet, key=time_compare)
    print("[INF] Number of filtered paths: ",len(learnSet))
    # Form the object goalsLearnedStructure
    goalsData = goal_pairs(areas,areasAxis,pathMat)
    return goalsData,pathMat,learnSet
