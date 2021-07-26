"""
"""
import sys, os, argparse, logging,random
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.io_parameters import read_and_set_parameters
from utils.io_trajectories import read_and_filter
from utils.io_trajectories import partition_train_test
from utils.manip_trajectories import observed_data
from utils.stats_trajectories import trajectory_arclength, trajectory_speeds
from gp_code.mGP_trajectory_prediction import mGP_trajectory_prediction
from utils.plotting import plotter, plot_path_samples
from utils.plotting import animate_multiple_predictions_and_goal_likelihood
from gp_code.likelihood import ADE
import numpy as np
import time

def main():
    # Parsing arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--coordinates', default='img',help='Type of coordinates (img or world)')
    parser.add_argument('--dataset_id', '--id',default='GCS',help='dataset id, GCS or EIF (default: GCS)')
    parser.add_argument('--log_level',type=int, default=20,help='Log level (default: 20)')
    parser.add_argument('--log_file',default='',help='Log file (default: standard output)')
    parser.add_argument('--no_pickle', dest='pickle',action='store_false')
    parser.set_defaults(pickle=True)
    args = parser.parse_args()
    if args.log_file=='':
        logging.basicConfig(format='%(levelname)s: %(message)s',level=args.log_level)
    else:
        logging.basicConfig(filename=args.log_file,format='%(levelname)s: %(message)s',level=args.log_level)

    # Read the areas file, dataset, and form the goalsLearnedStructure object
    imgGCS           = '../datasets/GC/reference.jpg'
    coordinates      = args.coordinates
    traj_dataset, goalsData, trajMat, __ = read_and_filter('GCS',coordinate_system=coordinates,use_pickled_data=args.pickle)
    train_set, test_set = partition_train_test(trajMat, training_ratio=0.8)
    
    # Selection of the kernel type
    kernelType  = "linePriorCombined"
    nParameters = 4

    # Read the kernel parameters from file
    goalsData.kernelsX = read_and_set_parameters("../parameters/linearpriorcombined20x20_GCS_img_x.txt",nParameters)
    goalsData.kernelsY = read_and_set_parameters("../parameters/linearpriorcombined20x20_GCS_img_y.txt",nParameters)

    part_num = 5
    nsamples = 5
    
    print('Test set shape:', test_set.shape)
        
    for i in range(test_set.shape[0]):
        for j in range(test_set.shape[1]):
            for tr in test_set[i][j]:
                # Prediction of single paths with a mixture model
                mgps     = mGP_trajectory_prediction(i,goalsData)
                arclen = trajectory_arclength(tr)
                tr_data = [tr[0],tr[1],arclen,tr[2]] # data = [X,Y,L,T]
                path_size = len(arclen)
                
                for k in range(1,part_num-1):
                    m = int((k+1)*(path_size/part_num))
                    observations, ground_truth = observed_data(tr_data,m)
                    gt = np.concatenate([np.reshape(tr[0][m:],(-1,1)), np.reshape(tr[1][m:],(-1,1))], axis=1)                  
                    # Multigoal prediction
                    likelihoods  = mgps.update(observations)
                    if(observations is None):
                        print('Observations is None!')
                    #filteredPaths= mgps.filter()
                    predictedXYVec,varXYVec = mgps.predict_trajectory(compute_sqRoot=True)
                    logging.debug('Generating samples')
                    paths = mgps.sample_paths(nsamples)
                    
                    pred_ade = []
                    for path in predictedXYVec:
                        pred_ade.append(ADE(gt, path))
                        
                    samples_ade = []
                    for path in paths:
                        samples_ade.append(ADE(gt, path))
                        
                    #print('Ground truth:')
                    #print(gt)
                    print('Prediction error:') 
                    print(pred_ade)
                    print('Sampled error:') 
                    print(samples_ade)
                    
#TODO: plot error
    
if __name__ == '__main__':
    main()

