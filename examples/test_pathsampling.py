"""
@author: karenlc
"""
from test_common import *
from gp_code.sGP_trajectory_prediction import sGP_trajectory_prediction

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
    imgGCS           = './datasets/GC/reference.jpg'
    coordinates      = args.coordinates
    img              = mpimg.imread(imgGCS)
    traj_dataset, goalsData, trajMat, __ = read_and_filter(args.dataset_id,use_pickled_data=args.pickle,coordinate_system=coordinates)
    # Selection of the kernel type
    kernelType = "linePriorCombined"
    nParameters = 4

    # Read the kernel parameters from file
    goalsData.kernelsX = read_and_set_parameters("parameters/linearpriorcombined20x20_GCS_img_x.txt",nParameters)
    goalsData.kernelsY = read_and_set_parameters("parameters/linearpriorcombined20x20_GCS_img_y.txt",nParameters)

    # Sampling 3 trajectories between all the pairs of goals
    allPaths = []
    for i in range(goalsData.goals_n):
        for j in range(i,goalsData.goals_n):
            if(i != j) and len(trajMat[i][j])>0 and goalsData.kernelsX[i][j].optimized is True:
                path  = trajMat[i][j][0]
                pathX, pathY, pathT = path
                pathL = trajectory_arclength(path)
                observations, __ = observed_data([pathX,pathY,pathL,pathT],2)
                if observations is None:
                    continue
                # The basic element here is this object, that will do the regression work
                gp = sGP_trajectory_prediction(i,j,goalsData)
                likelihood = gp.update(observations)
                # Generate samples
                predictedXY,varXY = gp.predict_path(compute_sqRoot=True)
                paths             = gp.sample_paths(3)
                allPaths          = allPaths + paths

    plot_path_samples(img,allPaths)


if __name__ == '__main__':
    main()
