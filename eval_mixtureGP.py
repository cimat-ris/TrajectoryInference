"""
"""
from test_common import *
from gp_code.mixture_gp import mixtureOfGPs

# Parsing arguments
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_id', '--id',default='GCS',help='dataset id, GCS or EIF (default: GCS)')
parser.add_argument('--log_level',type=int, default=20,help='Log level (default: 20)')
parser.add_argument('--log_file',default='',help='Log file (default: standard output)')
args = parser.parse_args()
if args.log_file=='':
    logging.basicConfig(format='%(levelname)s: %(message)s',level=args.log_level)
else:
    logging.basicConfig(filename=args.log_file,format='%(levelname)s: %(message)s',level=args.log_level)

# Read the areas file, dataset, and form the goalsLearnedStructure object
imgGCS           = './datasets/GC/reference.jpg'
coordinates      = args.coordinates

# Load test dataset

# Select kernel type

# Read kernel parameters 

# For each traj in test_dataset
    # create mgps
    # For each sub-part of the trajectory
        # get observations
        # update mgps
        # get most likely sample
        # compute ade and fde