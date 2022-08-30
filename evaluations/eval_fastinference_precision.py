"""
@author: jbhayet
"""
import sys, os, argparse, logging,random
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.io_parameters import read_and_set_parameters
from utils.manip_trajectories import observed_data
from utils.stats_trajectories import trajectory_arclength, trajectory_speeds
from gp_code.mGP_trajectory_prediction import mGP_trajectory_prediction
from utils.plotting import plotter, plot_path_samples
from utils.plotting import animate_multiple_predictions_and_goal_likelihood
from utils.io_trajectories import read_and_filter,partition_train_test
import numpy as np
import time,pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 24})

def main():
	# Parsing arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('--coordinates', default='world',help='Type of coordinates (img or world)')
	parser.add_argument('--dataset_id', '--id',default='GCS',help='dataset id, GCS or EIF (default: GCS)')
	parser.add_argument('--log_level',type=int, default=20,help='Log level (default: 20)')
	parser.add_argument('--log_file',default='',help='Log file (default: standard output)')
	parser.add_argument('--pickle', dest='pickle', action='store_true',help='uses previously pickled data')
	parser.set_defaults(pickle=False)

	args = parser.parse_args()
	if args.log_file=='':
		logging.basicConfig(format='%(levelname)s: %(message)s',level=args.log_level)
	else:
		logging.basicConfig(filename=args.log_file,format='%(levelname)s: %(message)s',level=args.log_level)

	# Read the areas file, dataset, and form the goalsLearnedStructure object
	dataset_name     = args.dataset_id
	traj_dataset, goalsData, trajMat, __ = read_and_filter(dataset_name,coordinate_system=args.coordinates,use_pickled_data=args.pickle)
	# Selection of the kernel type
	kernelType  = "linePriorCombined"
	img_bckgd   = './datasets/GC/reference.jpg'
	nSamples = 1
	# Partition test/train set
	pickle_dir='pickle'
	if args.pickle==False:
		train_trajectories_matrix, test_trajectories_matrix = partition_train_test(trajMat,training_ratio=0.2)
		pickle_out = open(pickle_dir+'/train_trajectories_matrix-'+args.dataset_id+'-'+args.coordinates+'.pickle',"wb")
		pickle.dump(train_trajectories_matrix, pickle_out, protocol=2)
		pickle_out.close()
		pickle_out = open(pickle_dir+'/test_trajectories_matrix-'+args.dataset_id+'-'+args.coordinates+'.pickle',"wb")
		pickle.dump(test_trajectories_matrix, pickle_out, protocol=2)
		pickle_out.close()
		"""**************    Learning GP parameters     **************************"""
		logging.info("Starting the learning phase")
		goalsData.sigmaNoise = 0.5
		goalsData.optimize_kernel_parameters(kernelType,train_trajectories_matrix)
		pickle_out = open(pickle_dir+'/goalsData-'+args.dataset_id+'-'+args.coordinates+'.pickle',"wb")
		pickle.dump(goalsData, pickle_out, protocol=2)
		pickle_out.close()
	else:
		pickle_in = open(pickle_dir+'/train_trajectories_matrix-'+args.dataset_id+'-'+args.coordinates+'.pickle', "rb")
		train_trajectories_matrix = pickle.load(pickle_in)
		pickle_in = open(pickle_dir+'/test_trajectories_matrix-'+args.dataset_id+'-'+args.coordinates+'.pickle', "rb")
		test_trajectories_matrix = pickle.load(pickle_in)
		pickle_in = open(pickle_dir+'/goalsData-'+args.dataset_id+'-'+args.coordinates+'.pickle',"rb")
		goalsData = pickle.load(pickle_in)
		pickle_in.close()

	"""**************    Learning GP parameters     **************************"""
	prop                 = 0.5
	precision_points     = []
	# Cycle over the goals
	for startG, endG in tqdm(np.ndindex(goalsData.goals_n,goalsData.goals_n),total=goalsData.goals_n*goalsData.goals_n):
		if goalsData.kernelsX[startG][endG] is not None and goalsData.kernelsY[startG][endG] is not None:
			# For each path from start to end
			for _path in test_trajectories_matrix[startG][endG]:
				# Get the path data
				pathL = trajectory_arclength(_path)
				# Total path length
				pathSize = len(_path)
				# Prediction of single paths with a mixture model
				mixture_model     = mGP_trajectory_prediction(startG,goalsData)

				# Take a proportion of the trajectory as observations
				knownN = int(prop*pathSize)
				logging.debug("--------------------------")
				# Monte Carlo
				observations, ground_truth = observed_data([_path[:,0],_path[:,1],pathL,_path[:,2]],knownN)
				"""Multigoal prediction test"""
				likelihoods, __  = mixture_model.update(observations,selection=False,consecutiveObservations=False)
				if likelihoods is None:
					continue
				else:
					logging.debug('{}'.format(likelihoods))
				filteredPaths= mixture_model.filter()
				predictedXYVec,varXYVec = mixture_model.predict_trajectory(compute_sqRoot=True)
				logging.debug('Generating samples')
				# Efficient sampling vs Slow sampling.
				# To evaluate the loss in precision, we use the same seed to compare the samples
				seed = int(time.time())
				st = time.process_time_ns()
				np.random.seed(seed)
				pathsEff, deltals,finalPositions  = mixture_model.sample_paths(nSamples,method=0)
				et = time.process_time_ns()
				print(finalPositions)
				print("Elapsed time: ",(et-st)/1000)

				np.random.seed(seed)
				pathsSlow, deltals,finalPositions = mixture_model.sample_paths(nSamples,method=1)
				et2 = time.process_time_ns()
				print(finalPositions)
				print("Elapsed time: ",(et2-et)/1000)

				np.random.seed(seed)
				pathsInterm, deltals,finalPositions = mixture_model.sample_paths(nSamples,method=2)
				et3 = time.process_time_ns()
				print(finalPositions)
				print("Elapsed time: ",(et3-et2)/1000)

				for idx in range(len(pathsEff)):
					error = np.sqrt(((pathsEff[idx][:,0:2] - pathsSlow[idx][:,0:2])**2).sum(axis=1).mean())
					precision_points.append(np.array([deltals[idx][0],error]))
				logging.debug("Error {}".format(error))
				if error>1:
					p = plotter(args.coordinates)
					p.set_background(img_bckgd)
					p.plot_scene_structure(goalsData)
					p.plot_path_samples_with_observations(observations,pathsEff,'blue')
					p.plot_path_samples_with_observations(observations,pathsSlow,'red')
					p.plot_path_samples_with_observations(observations,pathsInterm,'green')
					p.plot_startingpoints([observations[0]])
					p.plot_endingpoints(finalPositions)
					p.plot_likelihoods(goalsData,likelihoods)
					plt.show()

	precision_points  = np.array(precision_points)
	plt.figure()
	plt.scatter(precision_points[:,0],precision_points[:,1],alpha=0.3)
	plt.xlabel("Level of perturbation")
	plt.ylabel("Error (pixel)")
	plt.show()
if __name__ == '__main__':
	main()
