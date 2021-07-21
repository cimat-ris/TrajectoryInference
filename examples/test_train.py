"""
@author: karenlc
"""
from test_common import *
from utils.io_parameters import *
from utils.io_trajectories import read_and_filter,partition_train_test




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
    img_bckgd        = './datasets/GC/reference.jpg'
    img_bckgd        = './datasets/Edinburgh/edinburgh.jpg'
    dataset_name     = args.dataset_id
    coordinates      = args.coordinates

    traj_dataset, goalsData, trajMat, __ = read_and_filter(dataset_name,coordinate_system=coordinates,use_pickled_data=args.pickle)

    # Partition test/train set
    train_trajectories_matrix, test_trajectories_matrix = partition_train_test(trajMat)

    # Plot trajectories and structure
    showDataset = False
    if showDataset:
        for i in range(goalsData.goals_n):
            for j in range(goalsData.goals_n):
                if len(train_trajectories_matrix[i][j])>0:
                    p = plotter(title=f"Training trajectories from {i} to {j}")
                    p.set_background(img_bckgd)
                    p.plot_scene_structure(goalsData)
                    p.plot_paths(test_trajectories_matrix[i][j])
                    p.show()

    # Selection of the kernel type
    kernelType  = "linePriorCombined"

    """**************    Learning GP parameters     **************************"""
    logging.info("Starting the learning phase")
    goalsData.optimize_kernel_parameters(kernelType,train_trajectories_matrix)
    write_parameters(goalsData.kernelsX,"parameters/linearpriorcombined20x20_"+dataset_name+"_"+coordinates+"_x.txt")
    write_parameters(goalsData.kernelsY,"parameters/linearpriorcombined20x20_"+dataset_name+"_"+coordinates+"_y.txt")
    logging.info("End of the learning phase")

if __name__ == '__main__':
    main()
