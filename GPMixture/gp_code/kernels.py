from abc import ABCMeta, abstractmethod
import math as m
import numpy as np

# TODO: pass in a different way
# Standard deviation for the observation noise
# could we added it to the kernel class?
# nsigma = 7.50

# Returns two rowsxcolumns matrices:
# - the matrix of kernels with the default parameters
# - the matrix of kernel parameters (with default values)
def create_kernel_matrix(kerType, rows, columns):
    kerMatrix = []
    parameters = []
    # For goal i
    for i in range(rows):
        aux  = []
        auxP = []
        # For goal j
        for j in range(columns):
            kernel = set_kernel(kerType)
            theta  = kernel.get_parameters()
            aux.append(kernel)
            auxP.append(theta)
        kerMatrix.append(aux)
        parameters.append(auxP)
    return kerMatrix, parameters

# Set kernel: a function that creates a kernel with default parameters, given its type name
def set_kernel(name):
    if(name == "squaredExponential"):
        parameters = [80., 80.]  #{Covariance magnitude factor, Characteristic length}
        kernel = squaredExponentialKernel(parameters[0],parameters[1])
    elif(name == "combinedTrautman"):
        parameters = [60., 80., 80.]  #{Precision of the line constant, Covariance magnitude factor, Characteristic length}
        kernel = combinedTrautmanKernel(parameters[0],parameters[1],parameters[2])
    elif(name == "exponential"):
        parameters = [80., 80.]  #{Covariance magnitude factor, Characteristic length}
        kernel = exponentialKernel(parameters[0],parameters[1])
    elif(name == "gammaExponential"):
        parameters = [80., 80., 8.]  #{Covariance magnitude factor, Characteristic length, Gamma exponent}
        kernel = gammaExponentialKernel(parameters[0],parameters[1])
    elif(name == "rationalQuadratic"):
        parameters = [80.0,10., 5.] #{Covariance magnitude factor, Characteristic length, Alpha parameter}
        kernel = rationalQuadraticKernel(parameters[0],parameters[1])
    elif(name == "squaredExponentialAndNoise"):
        parameters = [80., 80.] #{Covariance magnitude factor, Characteristic length}
        kernel = squaredExponentialAndNoiseKernel(parameters[0],parameters[1])
    elif(name == "linePriorCombined"):
        parameters = [1.0,0.0,0.01,1.0, 80., 80.]  #{Mean slope, mean constant, Standard deviation slope, Standard deviation constant, Covariance magnitude factor, Characteristic length}
        kernel = linePriorCombinedKernel(parameters[0],parameters[1],parameters[2],parameters[3],parameters[4],parameters[5])
    return kernel

# Kronecker delta
def delta(x,y):
    if x == y:
        return 1
    else:
        return 0

# Abstract class for handling kernels
class Kernel:
    __metaclass__ = ABCMeta

    # Constructor
    def __init__(self):
        # Type of kernel
        self.type        = "generic"
        self.linearPrior = False

    # Overload the operator ()
    @abstractmethod
    def __call__(self,x,y): pass

    # Derivative with respect to y
    @abstractmethod
    def dkdy(self,x,y): pass

    def set_linear_prior(self,mS,mC,sS,sC):
        if self.linearPrior:
            # Linear prior parameters
            self.meanSlope     = mS
            self.meanConstant  = mC
            self.sigmaSlope    = sS
            self.sigmaConstant = sC

# Linear kernel
class linearKernel(Kernel):

    # Constructor
    def __init__(self, sigma_a, sigma_c):
        # Variance on the slope
        self.sigma_a   = sigma_a
        self.sigmaSq_a = sigma_a**2
        # Variance on the constant
        self.sigma_c   = sigma_c
        self.sigmaSq_c = sigma_c**2
        # Type of kernel
        self.type      = "linear"

    # Method to set parameters
    def setParameters(self,vec):
        # Variance on the slope
        self.sigma_a   = vec[0]
        self.sigmaSq_a = vec[0]**2
        # Variance on the constant
        self.sigma_c   = vec[1]
        self.sigmaSq_c = vec[1]**2

    # Method to get parameters
    def get_parameters(self):
        parameters = [self.sigma_a, self.sigma_c]
        return parameters

    # Overload the operator ()
    def __call__(self,l1,l2):
        return self.sigmaSq_a*((np.reshape(l1,(l1.shape[0],1))).dot((np.reshape(l2,(l2.shape[0],1))).transpose()))+self.sigmaSq_c

    # Derivative with respect to y
    def dkdy(self,x,y):
        return self.sigmaSq_a*x

# Kernel Trautman
class linearKernelTrautman(Kernel):
    # Constructor
    def __init__(self, gamma):
        self.gamma = gamma
        # Type of kernel
        self.type      = "linear"

    # Method to set parameters
    def setParameters(self,vec):
        self.gamma = vec[0]

    # Overload the operator ()
    def __call__(self,x,y):
        return x*y+(1./(self.gamma**2))

    # Derivative with respect to y
    def dkdy(self,x,y):
        return x

# Derived class: the squared exponential kernel
class squaredExponentialKernel(Kernel):
    # Constructor
    def __init__(self, sigmaSq, length):
        # Covariance magnitude factor
        self.sigmaSq      = sigmaSq
        # Characteristic length
        self.length       = length
        # Type of kernel
        self.type         = "squaredExponential"

    # Method to set parameters
    def setParameters(self,vec):
        # Covariance magnitude factor
        self.sigmaSq = vec[0]
        # Characteristic length
        self.length  = vec[1]

    # Overload the operator ()
    def __call__(self,x,y):
        return (self.sigmaSq**2)*m.exp(-1.*(x-y)**2/(2*self.length**2))

    # Derivative with respect to y
    def dkdy(self,x,y):
        return -(self.sigmaSq**2)*(y-x)/(2*self.length**2)*m.exp(-1.*(x-y)**2/(2*self.length**2))

# Matern kernel
class maternKernel(Kernel):
    # Constructor
    def __init__(self, sigmaSq, length):
        # Covariance magnitude factor
        self.sigmaSq      = sigmaSq
        # Characteristic length
        self.length       = length
        # Type of kernel
        self.type         = "matobservedern"
        self.sqrootof5    = m.sqrt(5)

    # Method to set parameters
    def setParameters(self,vec):
        # Covariance magnitude factor
        self.sigmaSq= vec[0]
        # Characteristic length
        self.length = vec[1]

    # Overload of operator ()
    def __call__(self,l1,l2):
        rn  = np.abs(l1[:, None] - l2[None, :])/self.length
        rn2 = rn**2
        return self.sigmaSq*(1. + self.sqrootof5*rn + 1.67*rn2)*np.exp(-self.sqrootof5*rn)

    # Derivative with respect to y
    def dkdy(self,x,y):
        rn  = m.fabs((x-y)/self.length)
        # To avoid overflow
        if rn>20.0:
            val = 0.0
        else:
            rn2 = rn**2
            rp  = m.copysign(1,y-x)/self.length
            val = -self.sigmaSq*rn*rp*1.67*(1. + self.sqrootof5*rn)*m.exp(-self.sqrootof5*rn)
        return val

# Matern kernel from Rasmussen
class maternRasmussenKernel(Kernel):
    # Constructor
    def __init__(self, sigmaSq, length):
        # Characteristic length
        self.length = length
        # Covariance magnitude factor
        self.sigmaSq= sigmaSq
        # Type of kernel
        self.type   = "maternRasmussen"

    # Method to set parameters
    def setParameters(self,vec):
        self.sigmaSq      = vec[0]
        # Characteristic length
        self.length       = vec[1]

    # Overload the operator ()
    def __call__(self,x,y):
        rn = m.fabs(x-y)/self.length
        val = self.sigmaSq*(1. + m.sqrt(3)*rn)*m.exp(-1.*m.sqrt(3)*rn)
        return val

# The combination used in the Trautman paper
class combinedTrautmanKernel(Kernel):
    # Constructor
    def __init__(self, gamma, sigmaSq, length, sigmaNoise):
        self.gamma = gamma
        # Covariance magnitude factor
        self.sigmaSq= sigmaSq
        # Characteristic length
        self.length = length
        self.linear = linearKernelTrautman(gamma)
        self.matern = maternKernel(sigmaSq,length)
        self.noise  = noiseKernel(sigmaNoise)
        # Type of kernel
        self.type      = "combinedTrautman"

    # Method to set parameters
    def setParameters(self,vec):
        self.gamma  = vec[0]
        # Covariance magnitude factor
        self.sigmaSq= vec[1]
        # Characteristic length
        self.length = vec[2]
        self.linear.setParameters(vec)
        mV = [vec[1], vec[2]]
        self.matern.setParameters(mV)

    # Overload the operator ()
    def __call__(self,x,y):
        return self.matern(x,y) + self.linear(x,y) + self.noise(x,y)

    # Method to get parameters
    def get_parameters(self):
        parameters = [self.gamma, self.sigmaSq, self.length]
        return parameters

    # Method to print parameters
    def print_parameters(self):
        print("combined kernel parameters\n gamma =",self.gamma,"\n s =",self.s,", l = ",self.length)

# Exponential kernel
class exponentialKernel(Kernel):
    # Constructor
    def __init__(self, sigmaSq, length):
        # Covariance magnitude factor
        self.sigmaSq   = sigmaSq
        # Characteristic length
        self.length    = length
        # Type of kernel
        self.type      = "exponential"

    # Method to set parameters
    def setParameters(self,vec):
        # Covariance magnitude factor
        self.sigmaSq = vec[0]
        # Characteristic length
        self.length  = vec[1]

    # Overload the operator ()
    def __call__(self,x,y):
        return self.sigmaSq*m.exp(-m.fabs(x-y)/self.length)

# Gamma exponential kernel
class gammaExponentialKernel(Kernel):
    # Constructor
    def __init__(self, sigmaSq, length, gamma):
        # Covariance magnitude factor
        self.sigmaSq = sigmaSq
        # Characteristic length
        self.length       = length
        # Gamma exponent
        self.gamma = gamma
        # Type of kernel
        self.type      = "gammaExponential"

    # Method to set parameters
    def setParameters(self,vec):
        # Covariance magnitude factor
        self.sigmaSq = vec[0]
        # Characteristic length
        self.length= vec[1]
        # Gamma exponent
        self.gamma = vec[2]

    # Overload the operator ()
    def __call__(self,x,y):
        rn = m.fabs(x-y)/self.length
        return self.sigmaSq*m.exp(-rn**self.gamma)

# Rational quadratic kernel
class rationalQuadraticKernel(Kernel):
    # Constructor
    def __init__(self, sigmaSq, length, alpha, sigmaNoise):
        # Covariance magnitude factor
        self.sigmaSq = sigmaSq
        # Characteristic length
        self.length  = length
        self.alpha   = alpha
        self.noise   = noiseKernel(sigmaNoise)
        # Type of kernel
        self.type    = "rationalQuadratic"

    # Method to set parameters
    def setParameters(self,vec):
        # Covariance magnitude factor
        self.sigmaSq = vec[0]
        # Characteristic length
        self.length  = vec[1]
        self.alpha   = vec[2]

    # Overload the operator ()
    def __call__(self,x,y):
        rn = m.fabs(x-y)/self.alpha*self.length
        return pow(1. + 0.5*(rn**2), -self.alpha) + self.noise(x,y)

# Combined kernel: squared exponential and noise
class squaredExponentialAndNoiseKernel(Kernel):
    # Constructor
    def __init__(self, sigmaSq, length, sigmaNoise):
        # Covariance magnitude factor
        self.sigmaSq      = sigmaSq
        # Characteristic length
        self.length      = length
        # Standard deviation of the noise
        self.sigmaNoise  = sigmaNoise
        # Type of kernel
        self.type    = "squaredExponentialAndNoise"

    # Method to set parameters
    def setParameters(self,vec):
        # Covariance magnitude factor
        self.sigmaSq = vec[0]
        # Characteristic length
        self.length  = vec[1]

    # Overload the operator ()
    def __call__(self,x,y):
        sqe   = squaredExponentialKernel(self.sigmaSq,self.length)
        noise = noiseKernel(self.sigmaNoise)
        return sqe(x,y) + noise(x,y)


# Combined kernel: exponential and noise
class exponentialAndNoiseKernel(Kernel):
    # Constructor
    def __init__(self, sigmaSq, length, sigmaNoise):
        self.exponential = exponentialKernel(sigmaSq,length)
        self.noise       = noiseKernel(sigmaNoise)
        # Type of kernel
        self.type    = "exponentialAndNoise"

    # Method to set parameters
    def setParameters(self,vec):
        self.exponential.setParameters(vec)

    # Overload the operator ()
    def __call__(self,x,y):
        return self.exponential(x,y) + self.noise(x,y)

# A combined kernel that considers a prior on the line parameters
class linePriorCombinedKernel(Kernel):
    # Constructor
    def __init__(self, meanSlope, meanConstant, sigmaSlope, sigmaConstant, sigmaSq, length):
        self.linearPrior   = True
        self.meanSlope     = meanSlope
        self.meanConstant  = meanConstant
        self.sigmaSlope    = sigmaSlope
        self.sigmaConstant = sigmaConstant
        # Covariance magnitude factor
        self.sigmaSq= sigmaSq
        # Characteristic length
        self.length = length
        self.linear = linearKernel(sigmaSlope, sigmaConstant)
        self.matern = maternKernel(sigmaSq,length)
        # Type of kernel
        self.type   = "linePriorCombined"

    # Method to set parameters
    def set_parameters(self,vec):
        # Linear prior parameters
        self.meanSlope     = vec[0]
        self.meanConstant  = vec[1]
        self.sigmaSlope    = vec[2]
        self.sigmaConstant = vec[3]
        # Covariance magnitude factor
        self.sigmaSq       = vec[4]
        # Characteristic length
        self.length        = vec[5]
        # Set the parameters of the sub-kernels
        self.linear.setParameters(vec[2:4])
        self.matern.setParameters(vec[4:6])

    # Method to set the optimizable parameters
    def set_optimizable_parameters(self,vec):
        # Covariance magnitude factor
        self.sigmaSq       = vec[0]
        # Characteristic length
        self.length        = vec[1]
        self.matern.setParameters(vec)

    # Overload the operator ()
    def __call__(self,l1,l2):
        K = self.matern(l1,l2)
        if self.linearPrior:
            K += self.linear(l1,l2)
        return K

    # Derivative with respect to y
    def dkdy(self,x,y):
        # TODO: if self.linearPrior:
        return self.matern.dkdy(x,y) + self.linear.dkdy(x,y)

    # Method to get parameters
    def get_parameters(self):
        parameters = [self.meanSlope, self.meanConstant, self.sigmaSlope, self.sigmaConstant, self.sigmaSq, self.length]
        return parameters

    # Method to get the optimizable parameters
    def get_optimizable_parameters(self):
        parameters = [self.sigmaSq, self.length]
        return parameters

    # Method to print parameters
    def print_parameters(self):
        print("combined kernel parameters\n gamma_a =",self.sigmaSlope,"\n gamma_0",self.sigmaConstant,"\n s =",self.sigmaSq,", l = ",self.length)
