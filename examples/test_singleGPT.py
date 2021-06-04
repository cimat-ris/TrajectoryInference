"""
Test single GP | Trautman's approach
"""
import random
from test_common import *
from gp_code.single_gp import singleGP
from gp_code.kernels import *

def main():
    # Parsing arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--coordinates', default='img',help='Type of coordinates (img or world)')
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
    traj_dataset, goalsData, trajMat, __ = read_and_filter(args.dataset_id,coordinate_system=coordinates,use_pickled_data=True)

    goalsData.kernelsX = create_kernel_matrix('combinedTrautman', goalsData.goals_n, goalsData.goals_n)
    goalsData.kernelsY = create_kernel_matrix('combinedTrautman', goalsData.goals_n, goalsData.goals_n)

    """
    print('[INF] Kernel parameters')
    for row in goalsData.kernelsX:
        for ker in row:
            ker.print_parameters()
    """
    """**********          Testing          ***********"""
    gi, gj, pathId = 0, 6, 5

    # Get the ground truth path
    path = trajMat[gi][gj][pathId]
    pathX, pathY, pathT = path
    # Total path length
    pathSize = len(pathX)

    # Prediction of single paths with single goals
    gp = singleGP(gi,gj,goalsData,'Trautman')

    # Divides the trajectory in part_num parts and infer the posterior over the remaining part
    part_num = 3
    for i in range(1,part_num-1):
        p = plotter()
        p.set_background(imgGCS)
        p.plot_scene_structure(goalsData)
        # Data we will suppose known
        knownN = int((i+1)*(pathSize/part_num))
        observations = observed_data(path,knownN)
        """Single goal prediction test"""
        # Update the GP with (real) observations
        start               = time.process_time()
        likelihood          = gp.update(observations)
        stop                = time.process_time()
        # Perform prediction
        logging.info('--- prediction---')
        predictedXY,varXY = gp.predict_path()
        logging.info('Plotting')
        logging.info('Likelihood: {}'.format(likelihood))
        # Plot the filtered version of the observations
        #p.plot_filtered(filteredX,filteredY)
        # Plot the prediction
        p.plot_prediction(observations,predictedXY,varXY)
        p.show()


if __name__ == '__main__':
    main()
