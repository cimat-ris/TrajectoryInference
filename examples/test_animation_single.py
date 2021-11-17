"""
@author: karenlc
"""
from test_common import *
from gp_code.sGP_trajectory_prediction import sGP_trajectory_prediction




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
    coordinates      = args.coordinates
    # Read the areas file, dataset, and form the goalsLearnedStructure object
    img_bckgd        = './datasets/GC/reference.jpg'
    #img_bckgd        = './datasets/Edinburgh/edinburgh.jpg'

    traj_dataset, goalsData, trajMat, __ = read_and_filter(args.dataset_id,coordinate_system=coordinates,use_pickled_data=args.pickle)

    # Selection of the kernel type
    kernelType  = "linePriorCombined"
    nParameters = 4

    # Read the kernel parameters from file
    goalsData.kernelsX = read_and_set_parameters("parameters/linearpriorcombined20x20_EIF_img_x.txt",nParameters)
    goalsData.kernelsY = read_and_set_parameters("parameters/linearpriorcombined20x20_EIF_img_y.txt",nParameters)

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
        logging.info('Selected goals: {:d} {:d}| path index:{:d}'.format(startG,endG,pathId))
    else:
        startG,endG = 0, 7
        pathId       = np.random.randint(0,len(trajMat[startG][endG]))
    # Kernels for this pair of goals
    kernelX = goalsData.kernelsX[startG][endG]
    kernelY = goalsData.kernelsY[startG][endG]
    # Get the ground truth path
    _path = trajMat[startG][endG][pathId]
    pathL = trajectory_arclength(_path)

    # Total path length
    pathSize = len(_path)

    # Test function: prediction of single paths with multiple goals
    gp = sGP_trajectory_prediction(startG,endG,goalsData)

    p = plotter()
    # For different sub-parts of the trajectory
    for knownN in range(10,pathSize-1):
        logging.info('--------------------------')
        p.set_background(img_bckgd)
        p.plot_scene_structure(goalsData)
        observations, ground_truth = observed_data([_path[:,0],_path[:,1],pathL,_path[:,2]],knownN)
        """Single goal prediction test"""
        logging.info('Updating observations')
        # Update the GP with (real) observations
        likelihood   = gp.update(observations)
        filteredPath = gp.filter()
        # Perform prediction
        predictedXY,varXY = gp.predict_trajectory()
        logging.info('Likelihood: {}'.format(likelihood))
        logging.info('Plotting')
        # Plot the filtered version of the observations
        # TODO: pbug with this plotting function
        # p.plot_filtered(filteredPath)
        # Plot the prediction
        p.plot_prediction(observations,predictedXY,varXY)
        # Plot the ground truth
        p.plot_ground_truth(ground_truth)
        p.pause(0.05)


if __name__ == '__main__':
    main()
