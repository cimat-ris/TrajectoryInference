import os,sys,pickle,argparse
from collections import OrderedDict
import numpy as np
import scipy
import torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../trajnetplusplustools')))

#import trajnetplusplustools
from trajnetplusplusbaselines.trajnetbaselines.lstm import LSTMPredictor

## Parallel Compute
import multiprocessing
from joblib import Parallel, delayed
from tqdm import tqdm

def process_scene(predictor, model_name, paths, scene_goal, args):
    ## For each scene, get predictions
    predictions = predictor(paths, scene_goal, n_predict=args.pred_length, obs_length=args.obs_length, modes=args.modes, args=args)
    return predictions

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

    args = parser.parse_args()

    np.seterr('ignore')

    ## Path to the data folder name to predict
    args.path = 'DATA_BLOCK/' + args.path + '/'

    ## Test_pred: Folders for saving model predictions
    args.path = args.path + 'test_pred/'

    if (not args.unimodal) and (not args.topk) and (not args.multimodal):
        args.unimodal = True # Compute unimodal metrics by default

    if args.topk:
        args.modes = 3

    if args.multimodal:
        args.modes = 20

    enable_col1 = True
    ## drop pedestrians that appear post observation
    def drop_post_obs(ground_truth, obs_length):
        obs_end_frame = ground_truth[0][obs_length].frame
        ground_truth = [track for track in ground_truth if track[0].frame < obs_end_frame]
        return ground_truth

    predictor = LSTMPredictor.load("lstm_directional_None.pkl")
    goal_flag = predictor.model.goal_flag

    # On CPU
    device = torch.device('cpu')
    predictor.model.to(device)

    total_scenes = 0
    average = 0
    final = 0
    gt_col = 0.
    pred_col = 0.
    neigh_scenes = 0
    topk_average = 0
    topk_final = 0
    all_goals = {}
    average_nll = 0

        ## Start writing in dataset/test_pred
        # for dataset in datasets:
            # Model's name
            # name = dataset.replace(args.path.replace('_pred', '') + 'test/', '')

            # Copy file from test into test/train_pred folder
            # print('processing ' + name)
            # if 'collision_test' in name:
            #    continue

            ## Filter for Scene Type
            # reader_tag = trajnetplusplustools.Reader(args.path.replace('_pred', '_private') + dataset + '.ndjson', scene_type='tags')
            # if args.scene_type != 0:
            #     filtered_scene_ids = [s_id for s_id, tag, s in reader_tag.scenes() if tag[0] == args.scene_type]
            # else:
            #     filtered_scene_ids = [s_id for s_id, _, _ in reader_tag.scenes()]

            # Read file from 'test'
            # reader = trajnetplusplustools.Reader(args.path.replace('_pred', '') + dataset + '.ndjson', scene_type='paths')
            ## Necessary modification of train scene to add filename (for goals)
            # scenes = [(dataset, s_id, s) for s_id, s in reader.scenes() if s_id in filtered_scene_ids]

            ## Consider goals
            ## Goal file must be present in 'goal_files/test_private' folder
            ## Goal file must have the same name as corresponding test file
            # if goal_flag:
            #    goal_dict = pickle.load(open('goal_files/test_private/' + dataset +'.pkl', "rb"))
            #    all_goals[dataset] = {s_id: [goal_dict[path[0].pedestrian] for path in s] for _, s_id, s in scenes}

            ## Get Goals
            # if goal_flag:
            #     scene_goals = [np.array(all_goals[filename][scene_id]) for filename, scene_id, _ in scenes]
            # else:
            #     scene_goals = [np.zeros((len(paths), 2)) for _, scene_id, paths in scenes]

            # print("Getting Predictions")
            # scenes = tqdm(scenes)
            ## Get all predictions in parallel. Faster!
            # pred_list = Parallel(n_jobs=12)(delayed(process_scene)(predictor, model_name, paths, scene_goal, args)
            #                                for (_, _, paths), scene_goal in zip(scenes, scene_goals))

            ## GT Scenes
            # reader_gt = trajnetplusplustools.Reader(args.path.replace('_pred', '_private') + dataset + '.ndjson', scene_type='paths')
            # scenes_gt = [s for s_id, s in reader_gt.scenes() if s_id in filtered_scene_ids]
            # total_scenes += len(scenes_gt)


if __name__ == '__main__':
    main()
