"""
"""
import os
import sys
import random
import logging
import argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np

from gp_code.mixture_gp import mixtureOfGPs
from utils.io_trajectories import read_and_filter
from utils.io_trajectories import partition_train_test
from utils.io_parameters import read_and_set_parameters
from utils.stats_trajectories import trajectory_arclength
from utils.manip_trajectories import observed_data
from utils.plotting import plotter, plot_path_samples



def main():
    # Parsing arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_id', '--id',default='GCS',help='dataset id, GCS or EIF (default: GCS)')
    parser.add_argument('--log_level',type=int, default=20,help='Log level (default: 20)')
    parser.add_argument('--log_file',default='',help='Log file (default: standard output)')
    args = parser.parse_args()
    if args.log_file=='':
        logging.basicConfig(format='%(levelname)s: %(message)s',level=args.log_level)
    else:
        logging.basicConfig(filename=args.log_file,format='%(levelname)s: %(message)s',level=args.log_level)
    
    # Read the areas file, dataset, and form the goalsLearnedStructure object
    imgGCS           = './datasets/GC/reference.jpg'
    coordinates      = args.coordinates
    
    # Load test dataset
    traj_dataset, goalsData, trajMat, __ = read_and_filter(args.dataset_id,coordinate_system=coordinates,use_pickled_data=True)
    train_set, test_set = partition_train_test(trajMat)
    
    # Select kernel type
    kernelType  = "linePriorCombined"
    nParameters = 4
    
    # Read the kernel parameters from file
    goalsData.kernelsX = read_and_set_parameters('parameters/linearpriorcombined20x20_GCS_img_x.txt',nParameters)
    goalsData.kernelsY = read_and_set_parameters('parameters/linearpriorcombined20x20_GCS_img_y.txt',nParameters)
    
    # Divides the trajectory in part_num parts and consider
    part_num = 5
    ade = np.empty()
    
    for i in range(part_num):
        ade[i] = []

    for i in range(test_set.shape[0]):
        for j in range(test_set.shape[1]):
            for tr in test_set[i][j]:
                # Prediction of single paths with a mixture model
                mgps     = mixtureOfGPs(i,goalsData)
                n_samples = 50
                arclen = trajectory_arclength(tr)
                path_size = len(arclen)
                
                tr_data = [tr[0],tr[1],arclen,tr[2]] # data = [X,Y,L,T]
                
                # For different sub-parts of the trajectory
                for k in range(1,part_num-1):
                    p = plotter()
                    p.set_background(imgGCS)
                    p.plot_scene_structure(goalsData)
                    
                    knownN = int((k+1)*(path_size/part_num)) #numero de datos conocidos
                    
                    observations, ground_truth = observed_data(tr_data,knownN)
                    """Multigoal prediction test"""
                    logging.info('Updating likelihoods')
                    likelihoods = mgps.update(observations)
                    logging.info('Performing prediction')
                    filteredPaths           = mgps.filter()
                    predictedXYVec,varXYVec = mgps.predict_trajectory()
                    logging.info('Plotting')
                    p.plot_multiple_predictions_and_goal_likelihood(observations,predictedXYVec,varXYVec,likelihoods)
                    logging.info('Goals likelihood:{}'.format(mgps.goalsLikelihood))
                    logging.info('Mean likelihood:{}'.format(mgps.meanLikelihood))
                    p.show()
                    # update mgps
                    # get most likely sample
                    # compute ade and fde
            
    
if __name__ == '__main__':
    main()