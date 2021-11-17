"""
Test GP misture | Trautman
"""
import random
from test_common import *
from gp_code.kernels import *
from gp_code.mGPt_trajectory_prediction import mGPt_trajectory_prediction
from utils.stats_trajectories import truncate


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
    traj_dataset, goalsData, trajMat, __ = read_and_filter(args.dataset_id,use_pickled_data=args.pickle)

    #I'm skipping the training for now

    goalsData.kernelsX = create_kernel_matrix('combinedTrautman', goalsData.goals_n, goalsData.goals_n)
    goalsData.kernelsY = create_kernel_matrix('combinedTrautman', goalsData.goals_n, goalsData.goals_n)

    """**********          Testing          ***********"""
    gi, gj, k = 0, 6, 5
    _path = trajMat[gi][gj][k]

    mgps     = mGPt_trajectory_prediction(gi,goalsData)
    nSamples = 5

    # Divides the trajectory in part_num parts and consider
    part_num = 3
    pathSize = len(path[0])

    # For different sub-parts of the trajectory
    for i in range(1,part_num-1):
        p = plotter()
        p.set_background("./datasets/GC/reference.jpg")
        p.plot_scene_structure(goalsData)

        knownN = int((i+1)*(pathSize/part_num)) #numero de datos conocidos
        observations = observed_data(path,knownN)
        observations, ground_truth = observed_data([_path[:,0],_path[:,1],_path[:,2]],knownN)

        """Multigoal prediction test"""
        logging.info('Updating likelihoods')
        likelihoods = mgps.update(observations)

        logging.info('Performing prediction')
        predictedXYVec,varXYVec = mgps.predict_path()
        #print('[INF] Plotting')
        #p.plot_multiple_predictions_and_goal_likelihood(path[0],path[1],knownN,goalsData.nGoals,likelihoods,predictedXYVec,varXYVec)

        logging.info('Goals likelihood:{}'.format(likelihoods))
        logging.info('Mean likelihood:{}'.format(mgps.meanLikelihood))
        logging.info('Generating samples')
        samplePaths = mgps.sample_paths(nSamples)
        p.plot_path_samples_with_observations(observations,samplePaths)
        p.show()


if __name__ == '__main__':
    main()
