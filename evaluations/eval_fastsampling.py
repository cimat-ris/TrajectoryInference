"""
@author: jbhayet
"""
import sys, os, argparse, logging,random
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.io_parameters import read_and_set_parameters
from utils.io_trajectories import read_and_filter
from utils.manip_trajectories import observed_data
from utils.stats_trajectories import trajectory_arclength, trajectory_speeds
from gp_code.mGP_trajectory_prediction import mGP_trajectory_prediction
from utils.plotting import plotter, plot_path_samples
from utils.plotting import animate_multiple_predictions_and_goal_likelihood
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
    imgGCS           = './datasets/GC/reference.jpg'
    coordinates      = args.coordinates
    traj_dataset, goalsData, trajMat, __ = read_and_filter('GCS',coordinate_system=coordinates,use_pickled_data=args.pickle)

    # Selection of the kernel type
    kernelType  = "linePriorCombined"
    nParameters = 4

    # Read the kernel parameters from file
    goalsData.kernelsX = read_and_set_parameters("parameters/linearpriorcombined20x20_GCS_img_x.txt",nParameters)
    goalsData.kernelsY = read_and_set_parameters("parameters/linearpriorcombined20x20_GCS_img_y.txt",nParameters)

    """**********          Testing          ***********"""
    # We give the start and ending goals
    startG,endG = 0, 7
    pathId       = np.random.randint(0,len(trajMat[startG][endG]))
    # Kernels for this pair of goals
    kernelX = goalsData.kernelsX[startG][endG]
    kernelY = goalsData.kernelsY[startG][endG]
    # Get the ground truth path
    path = trajMat[startG][endG][pathId]
    # Get the path data
    pathX, pathY, pathT = path
    pathL = trajectory_arclength(path)

    # Total path length
    pathSize = len(pathX)
    nSamples = 100
    # Prediction of single paths with a mixture model
    mgps     = mGP_trajectory_prediction(startG,goalsData)

    p = plotter()
    # For different sub-parts of the trajectory
    for knownN in range(5,pathSize-1):
        p.set_background(imgGCS)
        p.plot_scene_structure(goalsData,draw_ids=True)
        logging.info("--------------------------")
        tic = time.perf_counter()
        # Monte Carlo
        observations, ground_truth = observed_data([pathX,pathY,pathL,pathT],knownN)
        """Multigoal prediction test"""
        logging.info('Updating likelihoods')
        likelihoods  = mgps.update(observations)
        filteredPaths= mgps.filter()
        logging.info('Performing prediction')
        predictedXYVec,varXYVec = mgps.predict_trajectory(compute_sqRoot=True)
        logging.info('Generating samples')
        paths = mgps.sample_paths(nSamples)
        # Perform prediction
        toc = time.perf_counter()
        print('Time in s {}'.format(toc-tic))
        logging.info("Plotting")
        p.plot_path_samples_with_observations(observations,paths)
        logging.info("Goals likelihood {}".format(likelihoods))
        logging.info("Mean likelihood: {}".format(mgps.meanLikelihood))
        # Plot the ground truth
        p.plot_ground_truth(ground_truth)
        #p.pause(0.05)
        p.show()

if __name__ == '__main__':
    main()
