"""
@author: karenlc
"""
from test_common import *
from utils.plotting import *
from gp_code.kernels import *


# Evaluate covariance matrices on the interval [0,length]
def evaluateCovarianceMatrix(kernel,length):
    l = np.arange(0,length)
    return kernel(np.arange(0,l.size),np.arange(0,l.size))

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

kernel = exponentialKernel(parameters[2],parameters[3])
Cexp   = evaluateCovarianceMatrix(kernel,s)

# Display the covariance matrices
fg, axes = plt.subplots(1, 3, sharey=True)
axes[0].matshow(CSqe)
axes[1].matshow(CM)
axes[2].matshow(Cexp)
plt.show()
