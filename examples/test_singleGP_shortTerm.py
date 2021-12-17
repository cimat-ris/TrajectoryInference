"""
@author: karenlc
"""
import random
from test_common import *
from gp_code.sGPsT_trajectory_prediction import sGPsT_trajectory_prediction
import matplotlib.pyplot as plt

def main():
    # Parsing arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--coordinates', default='img',help='Type of coordinates (img or world)')
    parser.add_argument('--dataset_id', '--id',default='GCS',help='dataset id, GCS or EIF (default: GCS)')
    parser.add_argument('--log_level',type=int, default=20,help='Log level (default: 20)')
    parser.add_argument('--log_file',default='',help='Log file (default: standard output)')
    parser.add_argument('--pickle', dest='pickle', action='store_true',help='uses previously pickled data')
    parser.add_argument('--start', dest='start',type=int, default=0,help='Specify starting goal')
    parser.set_defaults(pickle=False)
    args = parser.parse_args()
    if args.log_file=='':
        logging.basicConfig(format='%(levelname)s: %(message)s',level=args.log_level)
    else:
        logging.basicConfig(filename=args.log_file,format='%(levelname)s: %(message)s',level=args.log_level)
    logging.getLogger('matplotlib.font_manager').disabled = True

    # Read the areas file, dataset, and form the goalsLearnedStructure object
    imgGCS           = './datasets/GC/reference.jpg'
    coordinates      = args.coordinates

    traj_dataset, goalsData, trajMat, filtered = read_and_filter(args.dataset_id,coordinate_system=coordinates,use_pickled_data=args.pickle)
    # Selection of the kernel type
    kernelType = "linePriorCombined"
    nParameters = 4

    # Read the kernel parameters from file
    goalsData.kernelsX = read_and_set_parameters('parameters/linearpriorcombined20x20_GCS_img_x.txt',nParameters)
    goalsData.kernelsY = read_and_set_parameters('parameters/linearpriorcombined20x20_GCS_img_y.txt',nParameters)

    """**********          Testing          ***********"""
    # We select a pair of starting and ending goals, and a trajectory id
    startG, nextG = args.start,-1
    flag = True
    while flag:
        nextG = random.randrange(goalsData.goals_n)
        if len(trajMat[startG][nextG]) > 0:
            pathId = random.randrange( len(trajMat[startG][nextG]))
            if len(trajMat[startG][nextG][pathId])>9:
                flag = False
    if len(trajMat[startG][nextG]) > 0 and  goalsData.kernelsX[startG][nextG].optimized is True:
        pathId       = np.random.randint(0,len(trajMat[startG][nextG]))
        logging.info("Selected goals: {} {} | path index: {}".format(startG,nextG,pathId))
    else:
        logging.error("This pair of goals have not optimized parameters. Aborting.")
        sys.exit()

    # Get the ground truth path
    _path   = trajMat[startG][nextG][pathId]
    _path_id=  int(_path[0][3])
    logging.info("ID: {:d}".format(_path_id))

    pathL = trajectory_arclength(_path)
    # Total path length
    pathSize = len(_path)

    # Prediction of single paths with single goals
    gp = sGPsT_trajectory_prediction(startG,nextG,goalsData)

    # Divides the trajectory in part_num parts and infer the posterior over the remaining part
    part_num = 10
    for i in range(1,part_num-1):
        p = plotter()
        logging.info('--------------------------')
        if coordinates=='img':
            p.set_background(imgGCS)
        p.plot_scene_structure(goalsData)
        # Data we will suppose known
        knownN = int((i+1)*(pathSize/part_num))
        observations, ground_truth = observed_data([_path[:,0],_path[:,1],pathL,_path[:,2]],knownN)
        # Starting/ending times of the observation
        etime = observations[-1,3]
        stime = observations[-4,3]
        neighbors_trajs = get_trajectories_given_time_interval(filtered, stime, etime)

        # Take the last 4 observations.
        # Caution, Social-GAN has been trained at 2.5fps. Here in GC we are at 1.25fps: So we interpolate!
        input= [[]]
        past = []
        for k in reversed(range(1,5)):
            # This position is interpolated!
            past.append(0.5*(observations[-k-1]+observations[-k]))
            past.append(observations[-k])
        past = np.array(past)

        neighbor_positions = []
        for neighbor_traj in neighbors_trajs:
            if len(neighbor_traj[neighbor_traj[:,2]==etime])==0:
                continue
            neighbor_position = neighbor_traj[(neighbor_traj[:,2]<=etime) & (neighbor_traj[:,2]>=stime)]
            # To avoid including the same trajctory that we are testing
            if neighbor_position[0,3]==_path_id:
                continue
            neighbor_positions.append(neighbor_position)

        """Single goal prediction test"""
        logging.info('Updating likelihoods')
        # Update the GP with (real) observations
        start                      = time.process_time()
        likelihood,preds_likelihood= gp.update(observations,consecutiveObservations=False)
        stop                       = time.process_time()
        filteredPath               = gp.filter()
        logging.info('CPU process time (update): {:.1f} [ms]'.format(1000.0*(stop-start)))
        start = stop
        # Perform prediction
        predictedXY,varXY = gp.predict_trajectory()
        stop              = time.process_time()
        logging.info('CPU process time (prediction): {:.1f} [ms]'.format((1000.0*(stop-start))))
        logging.info('Likelihood: {:f}'.format(likelihood))
        logging.info('Plotting')
        # Plot the filtered version of the observations
        p.plot_filtered([filteredPath])
        # Plot the prediction
        p.plot_prediction(observations,predictedXY,varXY)
        # Plot the ground truth
        p.plot_ground_truth(ground_truth)
        # Plot neighbors
        p.plot_neighbors(neighbor_positions)
        # Plot neighbors
        fig, ax = plt.subplots(2,1)
        ax[0].plot(predictedXY[:,2],predictedXY[:,0])
        ax[0].set_ylabel('x')
        ax[0].set_xlabel('l')
        ax[1].plot(predictedXY[:,2],predictedXY[:,1])
        ax[1].set_ylabel('y')
        ax[1].set_xlabel('l')
        p.show()


    # Same as above, with samples instead
    part_num = 10
    for i in range(1,part_num-1):
        p = plotter()
        p.set_background(imgGCS)
        p.plot_scene_structure(goalsData)
        # Data we will suppose known
        knownN            = int((i+1)*(pathSize/part_num))
        observations, __ = observed_data([pathX,pathY,pathL,pathT],knownN)
        """Single goal prediction test"""
        # Update the GP with (real) observations
        start      = time.process_time()
        likelihood = gp.update(observations)
        stop       = time.process_time()
        logging.info('CPU process time (update): {:.1f} [ms]'.format((1000.0*(stop-start))))
        start = stop
        # Form the predictive distribution
        predictedXY,varXY = gp.predict_path(compute_sqRoot=True)
        # Generate samples
        paths             = gp.sample_paths(10)
        # Plot samples and observations
        p.plot_path_samples_with_observations(observations,paths)
        stop       = time.process_time()
        logging.info('CPU process time (sampling): {:.1f} [ms]'.format(1000.0*(stop-start)))
        logging.info('Plotting')
        logging.info('Likelihood {:f}'.format(likelihood))
        p.show()


if __name__ == '__main__':
    main()
