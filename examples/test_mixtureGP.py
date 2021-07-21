"""
@author: karenlc
"""
from test_common import *
from gp_code.mGP_trajectory_prediction import mGP_trajectory_prediction



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

    traj_dataset, goalsData, trajMat, __ = read_and_filter(args.dataset_id,coordinate_system=coordinates,use_pickled_data=args.pickle)

    # Selection of the kernel type
    kernelType  = "linePriorCombined"
    nParameters = 4

    # Read the kernel parameters from file
    goalsData.kernelsX = read_and_set_parameters('parameters/linearpriorcombined20x20_GCS_img_x.txt',nParameters)
    goalsData.kernelsY = read_and_set_parameters('parameters/linearpriorcombined20x20_GCS_img_y.txt',nParameters)

    """******************************************************************************"""
    """**************    Testing                           **************************"""
    # We give the start and ending goals and the index of the trajectory to predict
    startG = 0
    nextG  = 6
    pathId = 3

    # Get the ground truth path
    _path = trajMat[startG][nextG][pathId]
    # Get the path data
    pathX, pathY, pathT = _path
    pathL = trajectory_arclength(_path)
    # Total path length
    pathSize = len(pathX)

    # Prediction of single paths with a mixture model
    mgps     = mGP_trajectory_prediction(startG,goalsData)
    nSamples = 50

    # Divides the trajectory in part_num parts and consider
    part_num = 5


    # For different sub-parts of the trajectory
    for i in range(1,part_num-1):
        p = plotter()
        p.set_background(imgGCS)
        p.plot_scene_structure(goalsData)

        knownN = int((i+1)*(pathSize/part_num)) #numero de datos conocidos
        observations, ground_truth = observed_data([pathX,pathY,pathL,pathT],knownN)
        """Multigoal prediction test"""
        logging.info('Updating likelihoods')
        likelihoods = mgps.update(observations)
        logging.info('Performing prediction')
        filteredPaths           = mgps.filter()
        predictedXYVec,varXYVec = mgps.predict_trajectory()
        logging.info('Plotting')
        p.plot_multiple_predictions_and_goal_likelihood(observations,predictedXYVec,varXYVec,likelihoods)
        logging.info('Goals likelihood:{}'.format(mgps._goals_likelihood))
        logging.info('Mean likelihood:{}'.format(mgps.meanLikelihood))
        p.show()


    # Again, with Monte Carlo
    for i in range(1,part_num-1):
        p = plotter()
        p.set_background(imgGCS)
        p.plot_scene_structure(goalsData)

        knownN = int((i+1)*(pathSize/part_num)) #numero de datos conocidos
        observations, ground_truth = observed_data([pathX,pathY,pathL,pathT],knownN)
        """Multigoal prediction test"""
        logging.info('Updating likelihoods')
        likelihoods = mgps.update(observations)
        logging.info('Performing prediction')
        predictedXYVec,varXYVec = mgps.predict_path(compute_sqRoot=True)
        logging.info('Goals likelihood {}'.format(mgps._goals_likelihood))
        logging.info('Mean likelihood: {}'.format(mgps.meanLikelihood))
        logging.info('Generating samples')
        paths = mgps.sample_paths(nSamples)
        logging.info('Plotting')
        p.plot_path_samples_with_observations(observations,paths)
        p.show()


if __name__ == '__main__':
    main()
