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
    traj_dataset, goalsData, trajMat, __ = read_and_filter('GCS',coordinate_system=coordinates,use_pickled_data=args.pickle)

    # Selection of the kernel type
    kernelType  = "linePriorCombined"
    nParameters = 4

    # Read the kernel parameters from file
    goalsData.kernelsX = read_and_set_parameters("parameters/linearpriorcombined20x20_GCS_img_x.txt",nParameters)
    goalsData.kernelsY = read_and_set_parameters("parameters/linearpriorcombined20x20_GCS_img_y.txt",nParameters)

    """**********          Testing          ***********"""
    # We give the start and ending goals
    randomPath = True
    if randomPath:
        flag = True
        while flag:
            startG, endG = random.randrange(goalsData.goals_n), random.randrange(goalsData.goals_n)
            if len(trajMat[startG][endG])>0 and goalsData.kernelsX[startG][endG].optimized is True:
                pathId = random.randrange( len(trajMat[startG][endG]) )
                flag = False
        logging.info("Selected goals: ({:d},{:d})| path index: {:d}".format(startG,endG,pathId))
    else:
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

    # Prediction of single paths with a mixture model
    mgps     = mGP_trajectory_prediction(startG,goalsData)

    p = plotter()
    # For different sub-parts of the trajectory
    for knownN in range(5,pathSize-1):
        logging.info("--------------------------")
        p.set_background(imgGCS)
        p.plot_scene_structure(goalsData)
        observations, ground_truth = observed_data([pathX,pathY,pathL,pathT],knownN)
        logging.info("Updating observations")
        # Update the GP with (real) observations
        likelihoods  = mgps.update(observations)
        filteredPaths= mgps.filter()
        # Perform prediction
        predictedXYVec,varXYVec = mgps.predict_trajectory()
        logging.info("Plotting")
        p.plot_multiple_predictions_and_goal_likelihood(observations,predictedXYVec,varXYVec,likelihoods)
        logging.info("Goals likelihood {}".format(likelihoods))
        logging.info("Mean likelihood: {}".format(mgps.meanLikelihood))
        # Plot the ground truth
        p.plot_ground_truth(ground_truth)
        p.pause(0.05)


if __name__ == '__main__':
    main()
