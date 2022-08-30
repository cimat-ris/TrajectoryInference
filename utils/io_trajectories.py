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
import logging

dataset_dirs = { 'GCS':'./datasets/GC/','EIF':'./datasets/Edinburgh/'}

# Get trajectories from files, and group them by pairs of goals
def get_traj_from_file(dataset_id, coordinate_system='img'):
    dataset_dir=dataset_dirs[dataset_id]
    if dataset_id=='GCS':
        traj_dataset= load_gcs(dataset_dir, coordinate_system=coordinate_system)
    else:
        if dataset_id=='EIF':
            traj_dataset= load_edinburgh(dataset_dir, coordinate_system=coordinate_system)
        else:
            logging.error('Cannot open this dataset')

    traj_set    = traj_dataset.get_trajectories()
    logging.info("Loaded {:s} set, length: {:03d} ".format(dataset_id,len(traj_set)))
    # Output will be a list of trajectories
    trajectories = []
    for tr in traj_set:
        # x, y, t: indices 0,1,4
        # p_id: index 5
        trajectories.append(tr[:,[0,1,4,5]])
    # Detect the trajectories that go between more than a pair of goals
    final_trajectories = []
    for tr in trajectories:
        # Split the trajectories
        final_trajectories.extend(break_multigoal_traj(tr, traj_dataset.goals_areas))
    logging.info("After handling goals, length: {:03d} ".format(len(final_trajectories)))
    return final_trajectories, traj_dataset.goals_areas

# Main function for reading the data from a dataset
def read_and_filter(dataset_id,use_pickled_data=False, pickle_dir='pickle', coordinate_system='img',filter=False):
    dataset_dir = dataset_dirs[dataset_id]
    if not use_pickled_data:
        # We process here multi-objective trajectories into sub-trajectories
        raw_trajectories, goals_areas = get_traj_from_file(dataset_id,coordinate_system=coordinate_system)
        # Dump trajectory dataset
        logging.info("Pickling data...")
        pickle_out = open(pickle_dir+'/trajectories-'+dataset_id+'-'+coordinate_system+'.pickle',"wb")
        pickle.dump(raw_trajectories, pickle_out, protocol=2)
        pickle_out.close()
        pickle_out = open(pickle_dir+'/goals-'+dataset_id+'-'+coordinate_system+'.pickle',"wb")
        pickle.dump(goals_areas, pickle_out, protocol=2)
        pickle_out.close()
    else:
        # Get trajectory dataset from pickle file
        logging.info("Unpickling...")
        pickle_in = open(pickle_dir+'/trajectories-'+dataset_id+'-'+coordinate_system+'.pickle',"rb")
        raw_trajectories = pickle.load(pickle_in)
        pickle_in = open(pickle_dir+'/goals-'+dataset_id+'-'+coordinate_system+'.pickle',"rb")
        goals_areas = pickle.load(pickle_in)

    logging.info("Assignment to goals.")
    # Get useful paths and split the trajectories into pairs of goals
    trajectories_matrix = separate_trajectories_between_goals(raw_trajectories, goals_areas)
    n = goals_areas.shape[0]
    s = 0
    for i in range(n):
        for j in range(n):
            s = s + len(trajectories_matrix[i][j])
    logging.info("Trajectories within goals {:d}".format(s))
    # Remove the trajectories that are either too short or too long
    if filter:
        filtered_trajectories_matrix, filtered_trajectories = filter_traj_matrix(trajectories_matrix)
    else:
        filtered_trajectories_matrix, filtered_trajectories =  trajectories_matrix, raw_trajectories
    s = 0
    for i in range(n):
        for j in range(n):
            s = s + len(filtered_trajectories_matrix[i][j])
    logging.info("Trajectories within goals (after filtering) {:d}".format(s))
    # Form the object goalsLearnedStructure
    goals_data = goal_pairs(goals_areas, filtered_trajectories_matrix,coordinate_system)
    return raw_trajectories, goals_data, filtered_trajectories_matrix, filtered_trajectories

# Partition the dataset between training and testing
def partition_train_test(trajectories_matrix,training_ratio=1.0):
    # Initialize a nRowsxnCols matrix with empty lists
    train_trajectories_matrix= np.empty(trajectories_matrix.shape,dtype=object)
    test_trajectories_matrix = np.empty(trajectories_matrix.shape,dtype=object)
    for i in range(trajectories_matrix.shape[0]):
        for j in range(trajectories_matrix.shape[1]):
            train_trajectories_matrix[i][j] = []
            test_trajectories_matrix[i][j] = []

    for i in range(trajectories_matrix.shape[0]):
        for j in range(trajectories_matrix.shape[1]):
            # Shuffle the list
            tr = trajectories_matrix[i][j]
            if len(tr)>0:
                idx      = np.arange(len(tr))
                np.random.shuffle(idx)
                train_idx= idx[0:int(training_ratio*len(tr))]
                test_idx = idx[int(training_ratio*len(tr)):]
                for k in train_idx:
                    train_trajectories_matrix[i][j].append(tr[k])
                for k in test_idx:
                    test_trajectories_matrix[i][j].append(tr[k])
    return train_trajectories_matrix, test_trajectories_matrix
