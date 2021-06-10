"""
"""
from test_common import *
from gp_code.mixture_gp import mixtureOfGPs
from utils.io_trajectories import read_and_filter
from utils.io_trajectories import partition_train_test



def main():
    # Parsing arguments
    parser = argparse.ArgumentParser()
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
    
    # Load test dataset
    traj_dataset, goalsData, trajMat, __ = read_and_filter(args.dataset_id,coordinate_system=coordinates,use_pickled_data=True)
    train_set, test_set = partition_train_test(trajMat)
    
    # Select kernel type
    kernelType  = "linePriorCombined"
    nParameters = 4
    
    # Read the kernel parameters from file
    goalsData.kernelsX = read_and_set_parameters('parameters/linearpriorcombined20x20_GCS_img_x.txt',nParameters)
    goalsData.kernelsY = read_and_set_parameters('parameters/linearpriorcombined20x20_GCS_img_y.txt',nParameters)

    
    # For each traj in test_dataset
    for i in range(test_set.shape[0]):
        for j in range(test_set.shape[1]):
            tr = trajectories_matrix[i][j]
            # Prediction of single paths with a mixture model
            mgps     = mixtureOfGPs(i,goalsData)
            nSamples = 50
        # create mgps
        # For each sub-part of the trajectory
            # get observations
            # update mgps
            # get most likely sample
            # compute ade and fde
            
    
if __name__ == '__main__':
    main()