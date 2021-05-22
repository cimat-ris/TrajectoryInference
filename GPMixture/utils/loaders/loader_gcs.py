# Author: Javad Amirian
# Email: amiryan.j@gmail.com

import json
from math import ceil

import os
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from utils.dataset_trajectories import TrajDataset


def image_to_world(p, Homog):
    pp = np.stack((p[:, 0], p[:, 1], np.ones(len(p))), axis=1)
    PP = np.matmul(Homog, pp.T).T
    P_normal = PP / np.repeat(PP[:, 2].reshape((-1, 1)), 3, axis=1)
    return P_normal[:, :2]


def load_gcs(path, **kwargs):
    traj_dataset = TrajDataset()
    raw_dataset = pd.DataFrame()
    annotation_path = os.path.join(path,"Annotations")
    file_list = sorted(os.listdir(annotation_path))
    raw_data_list = []  # the data to be converted into Pandas DataFrame
    coordinate_system = kwargs.get("coordinate_system", "img")
    selected_frames   = kwargs.get("frames", range(0, 120001))
    agent_id_incremental = 0

    # Read annotation files
    for annot_file in file_list:
        annot_file_full_path = os.path.join(annotation_path, annot_file)
        with open(annot_file_full_path, 'r') as f:
            annot_contents = f.read().split()

        agent_id = int(annot_file.replace('.txt', ''))
        agent_id_incremental += 1
        last_frame_id = -1

        for i in range(len(annot_contents) // 3):
            py = float(annot_contents[3 * i])
            px = float(annot_contents[3 * i + 1])
            frame_id = int(annot_contents[3 * i + 2])

            # there are trajectory files with non-continuous timestamps
            # they need to be counted as different agents
            if last_frame_id > 0 and (frame_id - last_frame_id) > 20:
                agent_id_incremental += 1
            last_frame_id = frame_id

            if selected_frames.start <= frame_id < selected_frames.stop:
                raw_data_list.append([frame_id, agent_id_incremental, px, py])

    csv_columns = ["frame_id", "agent_id", "pos_x", "pos_y"]
    raw_data_frame = pd.DataFrame(np.stack(raw_data_list), columns=csv_columns)

    raw_df_groupby = raw_data_frame.groupby("agent_id")
    trajs = [g for _, g in raw_df_groupby]

    tr0_ = trajs[0]
    tr1_ = trajs[1]

    for ii, tr in enumerate(trajs):
        if len(tr) < 2: continue
        # interpolate frames (2x up-sampling)
        interp_F = np.arange(tr["frame_id"].iloc[0], tr["frame_id"].iloc[-1], 10).astype(int)
        interp_X = interp1d(tr["frame_id"], tr["pos_x"], kind='linear')
        interp_X_= interp_X(interp_F)
        interp_Y = interp1d(tr["frame_id"], tr["pos_y"], kind='linear')
        interp_Y_= interp_Y(interp_F)
        agent_id = tr["agent_id"].iloc[0]
        raw_dataset = raw_dataset.append(pd.DataFrame({"frame_id": interp_F,
                                                       "agent_id": agent_id,
                                                       "pos_x": interp_X_,
                                                       "pos_y": interp_Y_}))
    raw_dataset = raw_dataset.reset_index()
    homog = [[4.97412897e-02, -4.24730883e-02, 7.25543911e+01],
             [1.45017874e-01, -3.35678711e-03, 7.97920970e+00],
             [1.36068797e-03, -4.98339188e-05, 1.00000000e+00]]
    if coordinate_system!="img":
        world_coords = image_to_world(raw_dataset[["pos_x", "pos_y"]].to_numpy(), homog)
        raw_dataset[["pos_x", "pos_y"]] = pd.DataFrame(world_coords * 0.8)

    # Copy columns
    traj_dataset.data[["frame_id", "agent_id", "pos_x", "pos_y"]] = \
        raw_dataset[["frame_id", "agent_id", "pos_x", "pos_y"]]
    traj_dataset.title = kwargs.get('title', "Grand Central")
    traj_dataset.data["scene_id"] = kwargs.get('scene_id', 0)
    traj_dataset.data["label"] = "pedestrian"
    fps = kwargs.get('fps', 25)
    # Read the areas data from the specified file
    goals_file                = os.path.join(path, "goals.csv")
    goals_data                = pd.read_csv(goals_file)
    traj_dataset.goals_areas  = goals_data.values[:,1:]
    # May have to transform goal areas coordinates
    if coordinate_system!="img":
        n_goals    = traj_dataset.goals_areas.shape[0]
        tmp = image_to_world(np.flip(np.reshape(traj_dataset.goals_areas[:,1:],(-1,2)),axis=1), homog)
        traj_dataset.goals_areas[:,1:] = np.reshape(np.flip(tmp,axis=1),(n_goals,-1))*0.8
    # post-process
    print("[INF] Post-process")
    sampling_rate = kwargs.get('sampling_rate', 1)
    use_kalman = kwargs.get('use_kalman', False)
    traj_dataset.postprocess(fps=fps, sampling_rate=sampling_rate, use_kalman=use_kalman)
    return traj_dataset


if __name__ == "__main__":
    import sys, os
    import matplotlib.pyplot as plt
    opentraj_root = sys.argv[1]
    path = os.path.join(opentraj_root, "datasets/GC/Annotation")
    traj_ds = load_gcs(path)
