"""
@author: jbhayet
"""
from test_common import *
from utils.loaders.loader_edinburgh import load_edinburgh
from utils.io_trajectories import read_and_filter,partition_train_test
from utils.manip_trajectories import separate_trajectories_between_goals
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

    # Read the areas file, dataset, and form the goalsLearnedStructure object
    img_bckgd        = './datasets/Edinburgh/edinburgh.jpg'
    coordinates      = args.coordinates
    # Edinburgh dataset
    dataset_dir = "./datasets/Edinburgh/"
    # Read the areas data from the specified file
    goals_file                = os.path.join(dataset_dir, "goals.csv")
    goals_data                = pd.read_csv(goals_file)
    goals_areas               = goals_data.values[:,1:]
    traj_dataset= load_edinburgh(dataset_dir)
    traj_dataset= traj_dataset.get_trajectories()
    trajectories= []
    for tr in traj_dataset:
        x, y, t = np.array(tr[:,0]), np.array(tr[:,1]), np.array(tr[:,4])
        trajectories.append(np.array([x,y,t]))
    logging.info("Loaded Edinburgh set, length: {:03d} ".format(len(trajectories)))
    trajectories_matrix, not_associated = separate_trajectories_between_goals(trajectories,goals_areas)
    goals_data = goal_pairs(goals_data.values[:,1:], trajectories_matrix)

    # Plot trajectories and structure
    showDataset = True
    p = plotter()
    if coordinates=='img':
        p.set_background(img_bckgd)
    logging.info("Unassociated trajectories: {:03d} ".format(len(not_associated)))
    p.plot_paths(not_associated)
    p.plot_scene_structure(goals_data,draw_ids=True)
    p.show()




if __name__ == '__main__':
    main()
