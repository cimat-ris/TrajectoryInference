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
from trajnetplusplusbaselines.trajnetbaselines.vae import VAEPredictor
from trajnetplusplusbaselines.trajnetbaselines.sgan import SGANPredictor
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
    logging.info('Loading data')
    traj_dataset, goalsData, trajMat, __ = read_and_filter(args.dataset_id,coordinate_system=coordinates,use_pickled_data=args.pickle)

    """******************************************************************************"""
    """**************    Testing                           **************************"""

    # Instanciate a predictor
    # This one is a single output predictor
    #predictor = LSTMPredictor.load("parameters/lstm_directional_None.pkl")
    #predictor = VAEPredictor.load("parameters/vae_directional_None.pkl")
    predictor = SGANPredictor.load("parameters/sgan_directional_None.pkl")
    # Important note: These predictors are learned at 2.5fps. Here we have data at 25fps
    scaleFPS = 5

    # On CPU
    device = torch.device('cpu')
    predictor.model.to(device)

    # We give the start and ending goals and the index of the trajectory to predict
    startG = args.start

    # Modes
    args.mode = 10

    # We sample 10 paths starting from this goal
    observations = []
    observations_used = []
    predictions  = []
    ground_truth = []

    for s in range(10):
        flag = True
        while flag:
            nextG = random.randrange(goalsData.goals_n)
            if len(trajMat[startG][nextG]) > 0:
                pathId = random.randrange( len(trajMat[startG][nextG]))
                if len(trajMat[startG][nextG][pathId][0])>9:
                    flag = False

        # Get the ground truth path
        _path = trajMat[startG][nextG][pathId]
        # Get the path data (x,y,t,l)
        pathX, pathY, pathT = _path
        pathL = trajectory_arclength(_path)
        # Total path length
        pathSize = len(pathX)
        # Divides the trajectory in part_num parts and consider a first part of it
        part_num = 5
        knownN = int(4*(pathSize/part_num)) #numero de datos conocidos
        obs, gt = observed_data([pathX,pathY,pathL,pathT],knownN)

        # Take the last 9 observations to be used as input by the predictor
        input= [[]]
        past = obs[-9*scaleFPS::scaleFPS]
        print(past.shape)
        if past.shape[0]<9:
            continue
        for i in range(9):
            input[0].append(TrackRow(past[i][2],10,past[i][0],past[i][1], None, 0))

        # Output is the following:
        # * Dictionnary of modes
        # * Each mode element is a list of agents. 0 is the agent of interest.
        preds = predictor(input, np.zeros((len(input), 2)), n_predict=args.pred_length, obs_length=args.obs_length, modes=args.mode, args=args)
        for l in range(args.mode):
            predictions.append(preds[l][0])
        ground_truth.append(gt)
        observations.append(obs)
        observations_used.append(past[:,0:2])

    #
    for s in range(len(ground_truth)):
        plt.figure(1)
        plt.plot(observations[s][:,0],observations[s][:,1],'b')
        plt.plot(observations_used[s][:,0],observations_used[s][:,1],'b+')
        plt.plot(ground_truth[s][:,0],ground_truth[s][:,1],'g')
        for l in range(args.mode):
            plt.plot(predictions[s*args.mode+l][:,0],predictions[s*args.mode+l][:,1],'r')
        plt.show()

if __name__ == '__main__':
    main()
