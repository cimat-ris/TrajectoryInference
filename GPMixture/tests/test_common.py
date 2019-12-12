import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import gp_code
from gp_code.goals_structure import goalsLearnedStructure
from utils.io_parameters import read_and_set_parameters
from utils.io_trajectories import read_and_filter
from utils.manip_trajectories import get_known_set
from utils.plotting import plot_prediction
from utils.plotting import plot_path_samples_with_observations
from utils.plotting import plot_multiple_predictions_and_goal_likelihood
from utils.plotting import animate_multiple_predictions_and_goal_likelihood
import pandas as pd
import numpy as np
import time
import matplotlib.image as mpimg
