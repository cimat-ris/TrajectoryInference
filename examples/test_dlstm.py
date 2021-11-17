import os,sys,pickle,argparse,logging
from collections import OrderedDict
from test_common import *
import numpy as np
import scipy
import torch
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../trajnetplusplustools')))
from utils.manip_trajectories import start_time, end_time, get_trajectories_given_time_interval, get_goal_of_point
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
    parser.add_argument('--obs_length', default=8, type=int,
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
    parser.add_argument('--filter', dest='filter',action='store_true',help='apply filtering')
    args = parser.parse_args()

    np.seterr('ignore')

    if args.log_file=='':
        logging.basicConfig(format='%(levelname)s: %(message)s',level=args.log_level)
    else:
        logging.basicConfig(filename=args.log_file,format='%(levelname)s: %(message)s',level=args.log_level)

    # Read the areas file, dataset, and form the goalsLearnedStructure object
    imgGCS           = './datasets/GC/reference.jpg'
    coordinates      = 'world'

    traj_dataset, goalsData, trajMat, filtered = read_and_filter(args.dataset_id,coordinate_system=coordinates,use_pickled_data=args.pickle,filter=args.filter)
    filtered.sort(key=start_time)

    """******************************************************************************"""
    """**************    Testing                           **************************"""

    # Instanciate a predictor
    #predictor = LSTMPredictor.load("parameters/lstm_directional_None.pkl")
    predictor = LSTMPredictor.load("parameters/sgan_directional_None.pkl")
    # On CPU
    device = torch.device('cpu')
    predictor.model.to(device)

    # We give the start and ending goals and the index of the trajectory to predict
    startG = args.start

    # Modes
    args.mode = 5

    # We sample 10 paths starting from this goal
    observations = []
    lastobservations = []
    predictions  = []
    ground_truth = []
    neighbors    = []
    tests = 1
    for s in range(tests):
        flag = True
        while flag:
            nextG = random.randrange(goalsData.goals_n)
            if len(trajMat[startG][nextG]) > 0:
                pathId = random.randrange( len(trajMat[startG][nextG]))
                if len(trajMat[startG][nextG][pathId])>9:
                    flag = False
        # Get the ground truth path
        _path = trajMat[startG][nextG][pathId]
        stime = start_time(_path)
        etime = end_time(_path)
        neighbors_trajs = get_trajectories_given_time_interval(filtered, stime, etime)
        # Get the path data
        pathL = trajectory_arclength(_path)
        # Total path length
        pathSize = len(_path)
        # Divides the trajectory in part_num parts and consider
        part_num = 5
        knownN = int(3*(pathSize/part_num)) #numero de datos conocidos
        obs, gt = observed_data([_path[:,0],_path[:,1],pathL,_path[:,2]],knownN)

        # Take the last 4 observations.
        # Caution, Social-GAN has been trained at 2.5fps. Here in GC we are at 1.25fps: So we interpolate!
        input= [[]]
        past = []
        for k in reversed(range(4)):
            # This position is interpolated!
            past.append(0.5*(obs[-k-1]+obs[-k]))
            past.append(obs[-k])
        past = np.array(past)

        neighbor_pos = []
        for neighbor_traj in neighbors_trajs:
            if len(neighbor_traj[neighbor_traj[:,2]==etime])==0:
                continue
            neighbor_pos.append(neighbor_traj[neighbor_traj[:,2]==etime])
        if len(neighbor_pos)>0:
            neighbor_pos = np.squeeze(np.array(neighbor_pos),axis=1)
        if (past.shape[0]<8):
            continue
        for i in range(8):
            input[0].append(TrackRow(past[i][2],10,past[i][0],past[i][1], None, 0))

        # Output is the following:
        # * Dictionnary of modes
        # * Each mode element is a list of agents. 0 is the agent of interest.
        preds = predictor(input, np.zeros((len(input), 2)), n_predict=args.pred_length, obs_length=args.obs_length, modes=args.mode, args=args)
        predictions.append(preds)
        # To visualize the whole set of observations
        observations.append(obs)
        # To visualize the observations used to perform the short-term prediction
        lastobservations.append(obs[-4:])
        ground_truth.append(gt)
        neighbors.append(neighbor_pos)
    #
    plt.figure(1)
    for s,pred in enumerate(predictions):
        for k in range(args.mode):
            mode =pred[k][0]
            plt.plot(mode[:,0],mode[:,1],'r')
        plt.plot(observations[s][:,0],observations[s][:,1],'b--')
        plt.plot(lastobservations[s][:,0],lastobservations[s][:,1],'b')
        if len(neighbor_pos)>0:
            plt.plot(neighbors[s][:,0],neighbors[s][:,1],'og')
    plt.show()

if __name__ == '__main__':
    main()
