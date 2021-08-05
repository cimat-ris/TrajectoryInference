"""
@author: karenlc
"""
import random
from test_common import *
from gp_code.sGP_trajectory_prediction import sGP_trajectory_prediction
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
    parser.add_argument('--end', dest='end',type=int, default=9,help='Specify ending goal')
    parser.set_defaults(pickle=False)
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
    kernelType = "linePriorCombined"
    nParameters = 4

    # Read the kernel parameters from file
    goalsData.kernelsX = read_and_set_parameters('parameters/linearpriorcombined20x20_GCS_img_x.txt',nParameters)
    goalsData.kernelsY = read_and_set_parameters('parameters/linearpriorcombined20x20_GCS_img_y.txt',nParameters)

    """**********          Testing          ***********"""
    # We select a pair of starting and ending goals, and a trajectory id
    gi, gj = args.start, args.end
    if len(trajMat[gi][gj]) > 0:
        pathId       = np.random.randint(0,len(trajMat[gi][gj]))
        logging.info("Selected goals: {} {} | path index: {}".format(gi,gj,pathId))

    # Get the ground truth path
    if goalsData.kernelsX[gi][gj].optimized is not True:
        logging.error("This pair of goals have not optimized parameters. Aborting.")
        sys.exit()

    path  = trajMat[gi][gj][pathId]
    pathX, pathY, pathT = path
    pathL = trajectory_arclength(path)
    # Total path length
    pathSize = len(pathX)

    # Prediction of single paths with single goals
    gp = sGP_trajectory_prediction(gi,gj,goalsData)

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
        observations, ground_truth = observed_data([pathX,pathY,pathL,pathT],knownN)
        """Single goal prediction test"""
        logging.info('Updating likelihoods')
        # Update the GP with (real) observations
        start        = time.process_time()
        likelihood   = gp.update(observations)
        stop         = time.process_time()
        filteredPath = gp.filter()
        logging.info('CPU process time (update): {:.1f} [ms]'.format(1000.0*(stop-start)))
        start = stop
        # Perform prediction
        predictedXY,varXY = gp.predict_trajectory()
        stop       = time.process_time()
        logging.info('CPU process time (prediction): {:.1f} [ms]'.format((1000.0*(stop-start))))
        logging.info('Likelihood: {:f}'.format(likelihood))
        logging.info('Plotting')
        # Plot the filtered version of the observations
        p.plot_filtered(filteredPath)
        # Plot the prediction
        p.plot_prediction(observations,predictedXY,varXY)
        # Plot the ground truth
        p.plot_ground_truth(ground_truth)
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
