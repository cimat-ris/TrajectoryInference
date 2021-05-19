import numpy as np
from gp_code.goal_pairs import goal_pairs
from utils.loaders.loader_ind import load_ind
from utils.loaders.loader_gcs import load_gcs
from utils.loaders.loader_edinburgh import load_edinburgh
from utils.manip_trajectories import break_multigoal_traj
from utils.manip_trajectories import separate_trajectories_between_goals
from utils.manip_trajectories import filter_traj_matrix
import pandas as pd
import pickle

# Get trajectories from files, and group them by pairs of goals
def get_traj_from_file(dataset_id, dataset_traj, coordinate_system='img'):
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
        # TODO: Should we check if there are repeated positions in the traj?
        trajectories.append([x,y,t])
    # Detect the trajectories that go between more than a pair of goals
    final_trajectories = []
    for tr in trajectories:
        # Split the trajectories
        final_trajectories.extend(break_multigoal_traj(tr, traj_dataset.goals_areas))
    print("[INF] After handling goals, length: {:03d} ".format(len(final_trajectories)))        
    return final_trajectories, traj_dataset.goals_areas

# new get_uncut_paths_from_file
# gets trajectories from the file without modifications
def get_complete_traj_from_file(dataset_id,dataset_traj,coordinate_system='img'):
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
def read_and_filter(dataset_id, trajectories_file, use_pickled_data=False, pickle_dir='pickle', coordinate_system='img'):
    if not use_pickled_data:
        # We process here multi-objective trajectories into sub-trajectories
        traj_dataset, goals_areas = get_traj_from_file(dataset_id,trajectories_file,coordinate_system=coordinate_system)
        # Dump trajectory dataset
        print("[INF] Pickling data...")
        pickle_out = open(pickle_dir+'/trajectories-'+dataset_id+'-'+coordinate_system+'.pickle',"wb")
        pickle.dump(traj_dataset, pickle_out, protocol=2)
        pickle_out.close()
        pickle_out = open(pickle_dir+'/goals-'+dataset_id+'-'+coordinate_system+'.pickle',"wb")
        pickle.dump(goals_areas, pickle_out, protocol=2)
        pickle_out.close()
    else:
        # Get trajectory dataset from pickle file
        print("[INF] Unpickling...")
        pickle_in = open(pickle_dir+'/trajectories-'+dataset_id+'-'+coordinate_system+'.pickle',"rb")
        traj_dataset = pickle.load(pickle_in)
        pickle_in = open(pickle_dir+'/goals-'+dataset_id+'-'+coordinate_system+'.pickle',"rb")
        goals_areas = pickle.load(pickle_in)

    print("[INF] Assignment to goals.")
    # Get useful paths and split the trajectories into pairs of goals
    trajMat, idx_out = separate_trajectories_between_goals(traj_dataset, goals_areas)
    n = goals_areas.shape[0]
    s = 0
    for i in range(n):
        for j in range(n):
            s = s + len(trajMat[i][j])
    print("[INF] Trajectories within goals ",s)
    # Remove the trajectories that are either too short or too long
    avgTrMat, avgTrajectories = filter_traj_matrix(trajMat)
    s = 0
    for i in range(n):
        for j in range(n):
            s = s + len(avgTrMat[i][j])
    print("[INF] Trajectories within goals (after filtering)",s)
    # Form the object goalsLearnedStructure
    goals_data = goal_pairs(goals_areas, avgTrMat)
    return traj_dataset, goals_data, avgTrMat, avgTrajectories, idx_out
