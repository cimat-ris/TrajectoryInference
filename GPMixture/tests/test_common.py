import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import gp_code
import gp_code.io_parameters
from gp_code.io_parameters import getUsefulPaths
from gp_code.io_parameters import read_and_set_parameters
from gp_code.goalsLearnedStructure import *
from utils.plotting import plot_prediction
from utils.plotting import plot_path_samples_with_observations
import utils
import pandas as pd
import numpy as np
import time
