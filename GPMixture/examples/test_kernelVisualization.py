"""
@author: karenlc
"""
from test_common import *
from utils.plotting import *
from gp_code.kernels import *


# Evaluate covariance matrices on the interval [0,length]
def evaluateCovarianceMatrix(kernel,length):
    l = np.arange(0,length)
    C = np.zeros((l.size,l.size),dtype=float)
    for i in range(0,length):
        for j in range(0,length):
            C[i][j]=kernel(i,j)
    return C

# Number of points to evaluate
s = 1000

# parameters[0]: For linear kernels, Standard deviation on slope.
# parameters[1]: for linear kernels, Standard deviation on constant.
# parameters[2]: Covariance multiplicative factor (gives the order of magnitude).
# parameters[3]: Characteristic length (gives radius of influence).
# parameters[4]: Standard deviation of noise.
parameters = [0.01, 2000, 500., 200., 1.0]
kernel = squaredExponentialKernel(parameters[2],parameters[3])
CSqe   = evaluateCovarianceMatrix(kernel,s)

kernel = maternKernel(parameters[2],parameters[3])
CM     = evaluateCovarianceMatrix(kernel,s)

kernel = linePriorCombinedKernel(parameters[0],parameters[1],parameters[2],parameters[3],parameters[4])
CCk    = evaluateCovarianceMatrix(kernel,s)

kernel = gammaExponentialKernel(parameters[2],parameters[3],0.8)
CG     = evaluateCovarianceMatrix(kernel,s)

kernel = squaredExponentialAndNoiseKernel(parameters[2],parameters[3],parameters[4])
CSqeN  = evaluateCovarianceMatrix(kernel,s)

kernel = exponentialAndNoiseKernel(parameters[2],parameters[3],parameters[4])
Cexp   = evaluateCovarianceMatrix(kernel,s)

# Display the covariance matrices
fg, axes = plt.subplots(2, 3, sharey=True)
axes[0][0].matshow(CSqe)
axes[0][1].matshow(CM)
axes[0][2].matshow(CG)

axes[1][0].matshow(CCk)
axes[1][1].matshow(CSqeN)
axes[1][2].matshow(Cexp)

plt.show()
