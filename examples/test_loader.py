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
    args = parser.parse_args()
    if args.log_file=='':
        logging.basicConfig(format='%(levelname)s: %(message)s',level=args.log_level)
    else:
        logging.basicConfig(filename=args.log_file,format='%(levelname)s: %(message)s',level=args.log_level)

    # InD dataset
    test_ind = False
    if test_ind:
        dataset_dir = "/home/jbhayet/opt/datasets/inD-dataset-v1.0/data/"
        for i in range(32):
            dataset_file= "{:02d}_tracks.csv".format(i)
            traj_dataset= load_ind(dataset_dir+dataset_file)
            traj_set    = traj_dataset.get_trajectories()
            logging.info("Loaded InD set {:02d}, length: {:03d} ".format(i,len(traj_set)))

    # GCS (Grand Central) dataset
    test_gcs = False
    if test_gcs:
        dataset_dir = "./datasets/GC/Annotation/"
        traj_dataset= load_gcs(dataset_dir)
        traj_set    = traj_dataset.get_trajectories()
        logging.info("Loaded gcs set, length: {:03d} ".format(len(traj_set)))

    # Edinburgh dataset
    test_edi = False
    if test_edi:
        dataset_dir = "./datasets/Edinburgh/annotations"
        traj_dataset= load_edinburgh(dataset_dir)
        traj_set    = traj_dataset.get_trajectories()
        logging.info("Loaded Edinburgh set, length: {:03d} ".format(len(traj_set)))

    if args.dataset_id=='GCS':
        img_bckgd        = './datasets/GC/reference.jpg'
    else:
        img_bckgd        = './datasets/Edinburgh/edinburgh.jpg'
    coordinates      =args.coordinates
    traj_dataset, goalsData, trajMat, __ = read_and_filter(args.dataset_id,coordinate_system=coordinates,use_pickled_data=True)

    # Plot trajectories and structure
    showDataset = True
    p = plotter()
    if coordinates=='img':
        p.set_background(img_bckgd)
    p.plot_scene_structure(goalsData)
    p.show()


if __name__ == '__main__':
    main()
