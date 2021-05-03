import numpy as np
from gp_code.goal_pairs import goal_pairs
from utils.loaders.loader_ind import load_ind
from utils.loaders.loader_gcs import load_gcs
from utils.loaders.loader_edinburgh import load_edinburgh
from utils.manip_trajectories import multigoal_trajectories
from utils.manip_trajectories import break_multigoal_traj
from utils.manip_trajectories import separate_trajectories_between_goals
from utils.manip_trajectories import filter_traj_matrix
import pandas as pd
import pickle

# Get trajectories from files, and group them by pairs of goals
def get_traj_from_file(dataset_id, dataset_traj, goals, coordinate_system='img'):
    # TODO: generalize to several datasets?
    if dataset_id=='GCS':
        traj_dataset= load_gcs(dataset_traj, coordinate_system=coordinate_system)
    else:
        if dataset_id=='edinburgh':
            traj_dataset= load_edinburgh(dataset_traj, coordinate_system=coordinate_system)
        else:
            print('[ERR] Cannot open this dataset')

    traj_set    = traj_dataset.get_trajectories()
    print("[INF] Loaded {:s} set, length: {:03d} ".format(dataset_id,len(traj_set)))
    # Output will be a list of trajectories
    trajectories = []
    for tr in traj_set:
        x, y, t = np.array(tr[:,0]), np.array(tr[:,1]), np.array(tr[:,4])
        # TODO: Should we check if there are repeat positions in the traj?
        trajectories.append([x,y,t])
    # Detect the trajectories that go between more than a pair of goals
    multigoal_traj = multigoal_trajectories(trajectories, goals)
    for tr in multigoal_traj:
        # Split these multiple-goal trajectories
        trajectories.extend(break_multigoal_traj(tr, goals) )
    return trajectories

# new get_uncut_paths_from_file
# gets trajectories from the file without modifications
def get_complete_traj_from_file(dataset_id,dataset_traj,goals,coordinate_system='img'):
    traj_dataset= load_gcs(dataset_traj, coordinate_system=coordinate_system)
    traj_set    = traj_dataset.get_trajectories()
    print("[INF] Loaded {:s} set, length: {:03d} ".format(dataset_id,len(traj_set)))
    trajectories = []
    # Get trajectories x,y,t
    for tr in traj_set:
        x, y, t = tr[:,0], tr[:,1], tr[:,4]
        trajectories.append([x,y,t])
    return trajectories

# Main function for reading the data from a dataset
def read_and_filter(dataset_id, areas_file, trajectories_file, use_pickled_data=False, pickle_dir='pickle'):
    # Read the areas data from the specified file
    data       = pd.read_csv(areas_file)
    goals_areas= data.values[:,1:]
    goals_n    = len(goals_areas)

    if not use_pickled_data:
        # We process here multi-objective trajectories into sub-trajectories
        traj_dataset = get_traj_from_file(dataset_id,trajectories_file, goals_areas)
        # Dump trajectory dataset
        print("[INF] Pickling data...")
        pickle_out = open(pickle_dir+'/trajectories.pickle',"wb")
        pickle.dump(traj_dataset, pickle_out, protocol=2)
        pickle_out.close()
    else:
        # Get trajectory dataset from pickle file
        print("[INF] Unpickling...")
        pickle_in = open(pickle_dir+'/trajectories.pickle',"rb")
        traj_dataset = pickle.load(pickle_in)

    print("[INF] Assignment to goals")
    # Get useful paths and split the trajectories into pairs of goals
    trajMat = separate_trajectories_between_goals(traj_dataset,goals_areas)
    # Remove the trajectories that are either too short or too long
    avgTrMat, avgTrajectories = filter_traj_matrix(trajMat,goals_n,goals_n)
    # Form the object goalsLearnedStructure
    goals_data = goal_pairs(goals_areas, avgTrMat)
    return traj_dataset, goals_data, avgTrMat, avgTrajectories
