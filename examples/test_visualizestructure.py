"""
@author: jbhayet
"""
from test_common import *
from utils.loaders.loader_ind import load_ind
from utils.loaders.loader_gcs import load_gcs
from utils.loaders.loader_edinburgh import load_edinburgh
from utils.io_trajectories import read_and_filter
import matplotlib.pyplot as plt
from utils.argparser import load_args_logs


def main():
    # Load arguments and set up logging
    args = load_args_logs()

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
