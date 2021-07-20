"""
@author: jbhayet
"""
import sys, os, argparse, logging,random
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.io_parameters import read_and_set_parameters
from utils.manip_trajectories import observed_data
from utils.stats_trajectories import trajectory_arclength, trajectory_speeds
from gp_code.mGP_trajectory_prediction import mGP_trajectory_prediction
from utils.plotting import plotter, plot_path_samples
from utils.plotting import animate_multiple_predictions_and_goal_likelihood
from utils.io_trajectories import read_and_filter,partition_train_test
import numpy as np
import time,pickle
from tqdm import tqdm

def main():
    # Parsing arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--coordinates', default='img',help='Type of coordinates (img or world)')
    parser.add_argument('--dataset_id', '--id',default='GCS',help='dataset id, GCS or EIF (default: GCS)')
    parser.add_argument('--log_level',type=int, default=20,help='Log level (default: 20)')
    parser.add_argument('--log_file',default='',help='Log file (default: standard output)')
    parser.add_argument('--pickle', dest='pickle', action='store_true',help='uses previously pickled data')
    parser.set_defaults(pickle=False)
    args = parser.parse_args()
    if args.log_file=='':
        logging.basicConfig(format='%(levelname)s: %(message)s',level=args.log_level)
    else:
        logging.basicConfig(filename=args.log_file,format='%(levelname)s: %(message)s',level=args.log_level)

    # Read the areas file, dataset, and form the goalsLearnedStructure object
    imgGCS           = './datasets/GC/reference.jpg'
    coordinates      = args.coordinates
    dataset_name     = args.dataset_id
    traj_dataset, goalsData, trajMat, __ = read_and_filter(dataset_name,coordinate_system=coordinates,use_pickled_data=args.pickle)

    # Selection of the kernel type
    kernelType  = "linePriorCombined"

    # Partition test/train set
    pickle_dir='pickle'
    if args.pickle==False:
        train_trajectories_matrix, test_trajectories_matrix = partition_train_test(trajMat,training_ratio=0.2)
        pickle_out = open(pickle_dir+'/train_trajectories_matrix-'+args.dataset_id+'-'+coordinates+'.pickle',"wb")
        pickle.dump(train_trajectories_matrix, pickle_out, protocol=2)
        pickle_out.close()
        pickle_out = open(pickle_dir+'/test_trajectories_matrix-'+args.dataset_id+'-'+coordinates+'.pickle',"wb")
        pickle.dump(train_trajectories_matrix, pickle_out, protocol=2)
        pickle_out.close()

        """**************    Learning GP parameters     **************************"""
        logging.info("Starting the learning phase")
        goalsData.optimize_kernel_parameters(kernelType,train_trajectories_matrix)
        pickle_out = open(pickle_dir+'/goalsData-'+args.dataset_id+'-'+coordinates+'.pickle',"wb")
        pickle.dump(goalsData, pickle_out, protocol=2)
        pickle_out.close()
    else:
        pickle_in = open(pickle_dir+'/train_trajectories_matrix-'+args.dataset_id+'-'+coordinates+'.pickle', "rb")
        train_trajectories_matrix = pickle.load(pickle_in)
        pickle_in = open(pickle_dir+'/test_trajectories_matrix-'+args.dataset_id+'-'+coordinates+'.pickle', "rb")
        test_trajectories_matrix = pickle.load(pickle_in)
        pickle_in = open(pickle_dir+'/goalsData-'+args.dataset_id+'-'+coordinates+'.pickle',"rb")
        goalsData = pickle.load(pickle_in)

    """**********          Testing          ***********"""
    timesEff  = []
    timesSlow = []
    ratios    = []
    for startG, endG in tqdm(np.ndindex(goalsData.goals_n,goalsData.goals_n),total=goalsData.goals_n*goalsData.goals_n):
            if goalsData.kernelsX[startG][endG] is not None and goalsData.kernelsY[startG][endG] is not None:
                for trajectory in test_trajectories_matrix[startG][endG]:
                    # Get the path data
                    pathX, pathY, pathT = trajectory
                    pathL = trajectory_arclength(trajectory)

                    # Total path length
                    pathSize = len(pathX)
                    nSamples = 100
                    # Prediction of single paths with a mixture model
                    mgps     = mGP_trajectory_prediction(startG,goalsData)

                    # Take half of the trajectory
                    knownN = pathSize//4
                    logging.debug("--------------------------")
                    # Monte Carlo
                    observations, ground_truth = observed_data([pathX,pathY,pathL,pathT],knownN)
                    """Multigoal prediction test"""
                    likelihoods  = mgps.update(observations)
                    if likelihoods is None:
                        continue
                    filteredPaths= mgps.filter()
                    predictedXYVec,varXYVec = mgps.predict_trajectory(compute_sqRoot=True)
                    logging.debug('Generating samples')
                    tic = time.perf_counter()
                    paths = mgps.sample_paths(nSamples,efficient=True)
                    toc = time.perf_counter()
                    timesEff.append(toc-tic)
                    tic = time.perf_counter()
                    paths = mgps.sample_paths(nSamples,efficient=False)
                    toc = time.perf_counter()
                    timesSlow.append(toc-tic)
                    ratios.append(timesEff[-1]/timesSlow[-1])
                    #        print("Largest deviations in x: {} and y: {}".format(np.max(np.abs(predictedX2-predictedX)),np.max(np.abs(predictedY2-predictedY))))
    print(np.mean(timesEff))
    print(np.mean(timesSlow))
    print(np.median(ratios))

if __name__ == '__main__':
    main()
