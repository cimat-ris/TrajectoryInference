"""
"""
import sys
sys.path.append('examples')
from test_common import *
from gp_code.mGP_trajectory_prediction import mGP_trajectory_prediction
import matplotlib.pyplot as plt

def save_values(file_name, values):
	with open(file_name+".txt", "w") as file:
		for row in values:
			s = " ".join(map(str, row))
			file.write(s+'\n')


def main():
	# Load arguments and set up logging
	args = load_args_logs()

	# Read the areas file, dataset, and form the goalsLearnedStructure object
	imgGCS           = 'datasets/GC/reference.jpg'

	# Selection of the kernel type
	kernelType  = "linePriorCombined"
	nParameters = 4

	# Partition test/train set
	pickle_dir='pickle'
	if args.pickle==False:
		train_trajectories_matrix, test_trajectories_matrix = partition_train_test(trajMat,training_ratio=0.2)
		pickle_out = open(pickle_dir+'/train_trajectories_matrix-'+args.dataset_id+'-'+args.coordinates+'.pickle',"wb")
		pickle.dump(train_trajectories_matrix, pickle_out, protocol=2)
		pickle_out.close()
		pickle_out = open(pickle_dir+'/test_trajectories_matrix-'+args.dataset_id+'-'+args.coordinates+'.pickle',"wb")
		pickle.dump(train_trajectories_matrix, pickle_out, protocol=2)
		pickle_out.close()

		logging.info("Starting the learning phase")
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

	# Read the kernel parameters from file
	goalsData.kernelsX = read_and_set_parameters("parameters/linearpriorcombined20x20",args.dataset_id,args.coordinates,'x')
	goalsData.kernelsY = read_and_set_parameters("parameters/linearpriorcombined20x20",args.dataset_id,args.coordinates,'y')
	# Set mode
	# mode = 'Trautman'
	mode = ""
	part_num = 4
	nsamples = 5

	ade = [[], [], [] ] # ade for 25%, 50% and 75% of observed data
	fde = [[], [], [] ]
	end_goal = [0,0,0]
	for i in range(3):#test_trajectories_matrix.shape[0]):
		for j in range(3):#test_trajectories_matrix.shape[1]):
			for tr in test_trajectories_matrix[i][j]:
				# Prediction of single paths with a mixture model
				mixture_model     = mGP_trajectory_prediction(i,goalsData)
				arclen = trajectory_arclength(tr)
				tr_data = [tr[:,0],tr[:,1],arclen,tr[:,2]] # data = [X,Y,L,T]

				path_size = len(arclen)
				if path_size < 10:
					continue

				for k in range(1,part_num):
					m = int((k)*(path_size/part_num))
					#if m<2:
					#    continue
					observations, ground_truth = observed_data(tr_data,m)
					gt = np.concatenate([np.reshape(tr[m:,0],(-1,1)), np.reshape(tr[m:,1],(-1,1))], axis=1)
					# Multigoal prediction
					likelihoods, __  = mixture_model.update(observations,selection=False,consecutiveObservations=False)
					if likelihoods is None:
						continue
					else:
						logging.debug('{}'.format(likelihoods))
					filteredPaths= mixture_model.filter()
					predictedXYVec,varXYVec = mixture_model.predict_trajectory(compute_sqRoot=True)
					logging.debug('Generating samples')
					paths,__,__,__ = mixture_model.sample_paths(nsamples)
					# Compare most likely goal and true goal
					if mixture_model.mostLikelyGoal == j:
						end_goal[k-1] += 1

					pred_ade = []
					for path in predictedXYVec:
						if path is  None:
							print('Prediction is None!')
						else:
							pred_ade.append(ADE(gt, path))

					samples_ade = []
					samples_fde = []
					for path in paths:
						if path is  None:
							print('Sample is None!')
						else:
							samples_ade.append(ADE(gt, path))
							samples_fde.append(FDE([gt[-1,0],gt[-1,1]], [path[-1,0],path[-1,1]] ))

					ade[k-1].append(round(min(samples_ade),3))
					fde[k-1].append(round(min(samples_fde),3))

	save_values('ade',ade)
	save_values('fde',fde)
	save_values('end_goal',[end_goal])

	print('traj mat shape:', test_trajectories_matrix.shape)
	print('Plotting mean ade')
	percent = [25,50,75]
	plt.figure()
	plt.plot(percent,[np.mean(np.array(ade[0])), np.mean(np.array(ade[1])), np.mean(np.array(ade[2]))])
	plt.xlabel('Percentage of observed data')
	plt.ylabel('Error in pixels')
	plt.show()
	#plt.savefig('ade.png')

	plt.figure()
	plt.plot(percent,[np.mean(np.array(fde[0])), np.mean(np.array(fde[1])), np.mean(np.array(fde[2]))])
	plt.xlabel('Percentage of observed data')
	plt.ylabel('Error in pixels')
	#plt.savefig('fde.png')
	plt.show()
	"""
	f, (ax1,ax2) = plt.subplots(1,2)
	ax1.plot(percent,[statistics.median(ade[0]), statistics.median(ade[1]), statistics.median(ade[2])])
	ax1.set_title('Median ADE')
	ax2.plot([25,50,75],[np.mean(np.array(ade[0])), np.mean(np.array(ade[1])), np.mean(np.array(ade[2]))])
	ax2.set_title('Mean ADE')
	ax1.set(xlabel='Percentage of observed data', ylabel='ADE error in pixels')
	"""


if __name__ == '__main__':
	main()
