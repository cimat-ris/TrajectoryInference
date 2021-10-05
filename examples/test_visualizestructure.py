"""
@author: jbhayet
"""
from test_common import *
from utils.loaders.loader_ind import load_ind
from utils.loaders.loader_gcs import load_gcs
from utils.loaders.loader_edinburgh import load_edinburgh
from utils.io_trajectories import read_and_filter
import matplotlib.pyplot as plt


def main():
    # Parsing arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--coordinates', default='img',help='Type of coordinates (img or world)')
    parser.add_argument('--dataset_id', '--id',default='GCS',help='dataset id, GCS or EIF (default: GCS)')
    parser.add_argument('--log_level',type=int, default=20,help='Log level (default: 20)')
    parser.add_argument('--log_file',default='',help='Log file (default: standard output)')
    parser.add_argument('--pickle', dest='pickle', action='store_true',help='Uses previously pickled data')
    parser.set_defaults(pickle=False)
    parser.add_argument('--no_draw_trajs', dest='no_draw_trajs', action='store_true',help='Draws trajectories samples')
    parser.set_defaults(no_draw_trajs=False)

    args = parser.parse_args()
    if args.log_file=='':
        logging.basicConfig(format='%(levelname)s: %(message)s',level=args.log_level)
    else:
        logging.basicConfig(filename=args.log_file,format='%(levelname)s: %(message)s',level=args.log_level)

    # Read the areas file, dataset, and form the goalsLearnedStructure object
    coordinates      = args.coordinates
    if args.dataset_id=='GCS':
        img_bckgd        = './datasets/GC/reference.jpg'
    else:
        img_bckgd        = './datasets/Edinburgh/edinburgh.jpg'
    coordinates      ='img'
    traj_dataset, goalsData, trajMat, __ = read_and_filter(args.dataset_id,coordinate_system=coordinates,use_pickled_data=args.pickle)
    logging.info("Number of trajectories: {:d}".format(len(traj_dataset)))
    # Plot trajectories and structure
    showDataset = True
    p = plotter()
    if coordinates=='img':
        p.set_background(img_bckgd)
    p.plot_scene_structure(goalsData,draw_ids=True)
    if args.no_draw_trajs==False:
        p.plot_paths_samples_gt(trajMat,n_samples=1)

    p.save("structure.pdf")
    p.show()


if __name__ == '__main__':
    main()
