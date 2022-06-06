"""
@author: jbhayet
"""
## TODO: repair

from test_common import *

def main():
    # Load arguments and set up logging
    args = load_args_logs()

    # Read the areas file, dataset, and form the goalsLearnedStructure object
    img_bckgd        = './datasets/Edinburgh/edinburgh.jpg'
    # Load the dataset
    raw_trajectories, goals_areas = get_traj_from_file('EIF',coordinate_system=args.coordinates)
    logging.info("Loaded Edinburgh set, length: {:03d} ".format(len(raw_trajectories)))
    paths_per_goals = separate_trajectories_between_goals(raw_trajectories,goals_areas)
    goals_data      = goal_pairs(goals_areas, paths_per_goals)

    # Plot trajectories and structure
    showDataset = True
    p = plotter()
    if args.coordinates=='img':
        p.set_background(img_bckgd)
    p.plot_scene_structure(goals_data,draw_ids=True)
    p.plot_paths_samples_gt(paths_per_goals, n_samples=4)
    p.show()


if __name__ == '__main__':
    main()
