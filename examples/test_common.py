import os
import sys
import random
import logging
import argparse
import pandas as pd
import numpy as np
import time,pickle
import matplotlib.image as mpimg

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import gp_code
from gp_code.likelihood import ADE,FDE
from gp_code.goal_pairs import goal_pairs
from utils.io_parameters import read_and_set_parameters
from utils.io_trajectories import read_and_filter,get_traj_from_file,partition_train_test
from utils.manip_trajectories import observed_data,get_trajectories_given_time_interval,separate_trajectories_between_goals
from utils.plotting import plotter, multiple_plotter, plot_path_samples
from utils.plotting import animate_multiple_predictions_and_goal_likelihood
from utils.stats_trajectories import trajectory_arclength, trajectory_speeds
from utils.argparser import load_args_logs
from utils.loaders.loader_edinburgh import load_edinburgh
