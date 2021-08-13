import os,sys,pickle,argparse,logging
from collections import OrderedDict
from test_common import *
import numpy as np
import scipy
import torch
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../trajnetplusplustools')))

#import trajnetplusplustools
from trajnetplusplusbaselines.trajnetbaselines.lstm import LSTMPredictor
from trajnetplusplustools.reader import Reader
from trajnetplusplustools.data import TrackRow
## Parallel Compute
import multiprocessing
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='trajdata',
                        help='directory of data to test')
    parser.add_argument('--model', nargs='+',
                        help='relative path to saved model')
    parser.add_argument('--obs_length', default=9, type=int,
                        help='observation length')
    parser.add_argument('--pred_length', default=12, type=int,
                        help='prediction length')
    parser.add_argument('--unimodal', action='store_true',
                        help='provide unimodal evaluation')
    parser.add_argument('--topk', action='store_true',
                        help='provide topk evaluation')
    parser.add_argument('--multimodal', action='store_true',
                        help='provide multimodal nll evaluation')
    parser.add_argument('--scene_type', default=0, type=int,
                        choices=(0, 1, 2, 3, 4),
                        help='type of scene to evaluate')
    parser.add_argument('--thresh', default=0.0, type=float,
                        help='noise thresh')
    parser.add_argument('--ped_type', default='primary',
                        help='type of ped to add noise to')
    parser.add_argument('--normalize_scene', action='store_true',
                        help='augment scenes')
    parser.add_argument('--dataset_id', '--id',default='GCS',help='dataset id, GCS or EIF (default: GCS)')
    parser.add_argument('--log_level',type=int, default=20,help='Log level (default: 20)')
    parser.add_argument('--log_file',default='',help='Log file (default: standard output)')
    parser.add_argument('--pickle', dest='pickle', action='store_true',help='uses previously pickled data')
    parser.add_argument('--start', dest='start',type=int, default=0,help='Specify starting goal')

    args = parser.parse_args()

    np.seterr('ignore')

    if args.log_file=='':
        logging.basicConfig(format='%(levelname)s: %(message)s',level=args.log_level)
    else:
        logging.basicConfig(filename=args.log_file,format='%(levelname)s: %(message)s',level=args.log_level)

    # Read the areas file, dataset, and form the goalsLearnedStructure object
    imgGCS           = './datasets/GC/reference.jpg'
    coordinates      = 'world'

    traj_dataset, goalsData, trajMat, __ = read_and_filter(args.dataset_id,coordinate_system=coordinates,use_pickled_data=args.pickle)

    # Selection of the kernel type
    kernelType  = "linePriorCombined"
    nParameters = 4

    # Read the kernel parameters from file
    goalsData.kernelsX = read_and_set_parameters('parameters/linearpriorcombined20x20_GCS_img_x.txt',nParameters)
    goalsData.kernelsY = read_and_set_parameters('parameters/linearpriorcombined20x20_GCS_img_y.txt',nParameters)

    """******************************************************************************"""
    """**************    Testing                           **************************"""
    # We give the start and ending goals and the index of the trajectory to predict
    startG = args.start
    flag = True
    while flag:
        nextG = random.randrange(goalsData.goals_n)
        if len(trajMat[startG][nextG]) > 0:
            pathId = random.randrange( len(trajMat[startG][nextG]))
            flag = False

    # Get the ground truth path
    _path = trajMat[startG][nextG][pathId]
    # Get the path data
    pathX, pathY, pathT = _path
    pathL = trajectory_arclength(_path)
    # Total path length
    pathSize = len(pathX)

    # Divides the trajectory in part_num parts and consider
    part_num = 5
    knownN = int(3*(pathSize/part_num)) #numero de datos conocidos
    observations, ground_truth = observed_data([pathX,pathY,pathL,pathT],knownN)

    # Take the last 9 observations
    past = observations[-9:]
    rows = [[]]
    logging.info("Observed:")
    logging.info("{}".format(past[:,0:2]))
    for i in range(9):
        row = TrackRow(past[i][2],10,past[i][0],past[i][1], None, 0)
        rows[0].append(row)

    if (not args.unimodal) and (not args.topk) and (not args.multimodal):
        args.unimodal = True # Compute unimodal metrics by default

    if args.topk:
        args.modes = 3

    if args.multimodal:
        args.modes = 20

    predictor = LSTMPredictor.load("lstm_directional_None.pkl")
    goal_flag = predictor.model.goal_flag

    # On CPU
    device = torch.device('cpu')
    predictor.model.to(device)

    predictions = predictor(rows, np.zeros((len(rows), 2)), n_predict=args.pred_length, obs_length=args.obs_length, modes=3, args=args)
    logging.info("Predicted:")
    logging.info("{}".format(predictions[0][0]))

    #
    plt.figure(1)
    plt.plot(predictions[0][0][:,0],predictions[0][0][:,1],'r')
    plt.plot(past[:,0],past[:,1],'b')
    plt.show()

if __name__ == '__main__':
    main()
