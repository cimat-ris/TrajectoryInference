"""
@author: karenlc
"""
from test_common import *
from utils.stats_trajectories import tr_histogram




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
    coordinates      = args.coordinates
    dataset          = args.dataset_id
    traj_dataset, goalsData, trajMat, __ = read_and_filter(dataset,coordinate_system=coordinates,use_pickled_data=args.pickle)


    # We give the start and ending goals
    startG = 0
    nextG  = 6
    tr_histogram(trajMat[startG][nextG])


if __name__ == '__main__':
    main()
