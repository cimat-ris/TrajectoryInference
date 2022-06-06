"""
@author: karenlc
"""
from test_common import *
from gp_code.mGP_trajectory_prediction import mGP_trajectory_prediction

def main():

    # Load arguments and set up logging
    args = load_args_logs()

    # Read the areas file, dataset, and form the goalsLearnedStructure object
    imgGCS           = './datasets/GC/reference.jpg'
    traj_dataset, goalsData, trajMat, __ = read_and_filter('GCS',coordinate_system=args.coordinates,use_pickled_data=args.pickle)

    # Read the kernel parameters from file
    goalsData.kernelsX = read_and_set_parameters('parameters/linearpriorcombined20x20',args.dataset_id,args.coordinates,'x')
    goalsData.kernelsY = read_and_set_parameters('parameters/linearpriorcombined20x20',args.dataset_id,args.coordinates,'y')
    goalsData.sigmaNoise = 600.0
    """**********          Testing          ***********"""
    # We give the start and ending goals
    randomPath = True
    if randomPath:
        flag = True
        while flag:
            startG, endG = random.randrange(goalsData.goals_n), random.randrange(goalsData.goals_n)
            if len(trajMat[startG][endG])>0 and goalsData.kernelsX[startG][endG].optimized is True:
                pathId = random.randrange( len(trajMat[startG][endG]) )
                flag = False
        logging.info("Selected goals: ({:d},{:d})| path index: {:d}".format(startG,endG,pathId))
    else:
        startG,endG = 0, 7
        pathId       = np.random.randint(0,len(trajMat[startG][endG]))
    # Kernels for this pair of goals
    kernelX = goalsData.kernelsX[startG][endG]
    kernelY = goalsData.kernelsY[startG][endG]
    # Get the ground truth path
    _path = trajMat[startG][endG][pathId]
    # Get the path data
    pathL = trajectory_arclength(_path)
    # Total path length
    pathSize = len(_path)

    # Prediction of single paths with a mixture model
    mgps     = mGP_trajectory_prediction(startG,goalsData)

    # For different sub-parts of the trajectory
    for knownN in range(5,pathSize-1):
        p = plotter()
        logging.info("--------------------------")
        p.set_background(imgGCS)
        p.plot_scene_structure(goalsData,draw_ids=True)
        observations, ground_truth = observed_data([_path[:,0],_path[:,1],pathL,_path[:,2]],knownN)
        logging.info("Updating observations")
        # Update the GP with (real) observations
        likelihoods, __  = mgps.update(observations,consecutiveObservations=False)
        filteredPaths= mgps.filter()
        # Perform prediction
        predictedXYVec,varXYVec = mgps.predict_trajectory()
        logging.info("Plotting")
        p.plot_multiple_predictions_and_goal_likelihood(observations,predictedXYVec,varXYVec,likelihoods)
        logging.info("Goals likelihood {}".format(likelihoods))
        logging.info("Mean likelihood: {}".format(mgps.meanLikelihood))
        # Plot the ground truth
        p.plot_ground_truth(ground_truth)
        p.show()

if __name__ == '__main__':
    main()
