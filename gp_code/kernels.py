from abc import ABCMeta, abstractmethod
import math as m
import numpy as np
import logging

# Returns two rowsxcolumns matrices:
# - the matrix of kernels with the default parameters
def create_kernel_matrix(kerType, rows, columns):
    matrix = np.empty((rows,columns),dtype=object)
    for i in range(rows):
        for j in range(columns):
            matrix[i][j] = set_kernel(kerType)
    return matrix

# Set kernel: a function that creates a kernel with default parameters, given its type
def set_kernel(type_):
    if(type_ == "combinedTrautman"):
        parameters = [60., 80., 80.]  #{Precision of the line constant, Covariance magnitude factor, Characteristic length}
        kernel = combinedTrautmanKernel(parameters[0],parameters[1],parameters[2])
        kernel.optimized = False
    elif(type_ == "linePriorCombined"):
        parameters = [1.0,0.0,0.01,1.0, 100., 50.]  #{Mean slope, mean constant, Standard deviation slope, Standard deviation constant, Covariance magnitude factor, Characteristic length}
        kernel = linePriorCombinedKernel(parameters[0],parameters[1],parameters[2],parameters[3],parameters[4],parameters[5])
        kernel.optimized = False
    elif(type_ == "squaredExponential"):
        parameters = [80., 80.]  #{Covariance magnitude factor, Characteristic length}
        kernel = squaredExponentialKernel(parameters[0],parameters[1])
        kernel.optimized = False
    elif(type_ == "exponential"):
        parameters = [80., 80.]  #{Covariance magnitude factor, Characteristic length}
        kernel = exponentialKernel(parameters[0],parameters[1])
        kernel.optimized = False
    return kernel

# Abstract class for handling kernels
class Kernel:
    __metaclass__ = ABCMeta

    # Constructor
    def __init__(self):
        # Type of kernel
        self.type        = "generic"
        self.linearPrior = False
        self.optimized   = False

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
    def set_parameters(self,vec):
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
    def set_parameters(self,vec):
        self.gamma = vec[0]

    # Overload the operator ()
    def __call__(self,x,y):
        return (np.reshape(x,(x.shape[0],1))).dot((np.reshape(y,(y.shape[0],1))).transpose()) +(1./(self.gamma**2))

    # Derivative with respect to y
    def dkdy(self,x,y):
        return x


# Matern kernel
class maternKernel(Kernel):
    # Constructor
    def __init__(self, sigmaSq, length):
        # Covariance magnitude factor
        self.sigmaSq      = sigmaSq
        # Characteristic length
        self.length       = length
        # Type of kernel
        self.type         = "matern"
        self.sqrootof5    = m.sqrt(5)

    # Method to set parameters
    def set_parameters(self,vec):
        # Covariance magnitude factor
        self.sigmaSq= vec[0]
        # Characteristic length
        self.length = vec[1]

    # Method to set parameters
    def set_optimizable_parameters(self,vec):
        # Covariance magnitude factor
        self.sigmaSq= vec[0]
        # Characteristic length
        self.length = vec[1]

    # Overload of operator ()
    def __call__(self,l1,l2):
        l = l1[:, None] - l2[None, :]
        rn  = np.abs(l)/self.length
        if rn.min()<0:
            logging.error("Numerical problem: {} {}".format(rn.min(),self.length))
        rn2 = rn**2
        return self.sigmaSq*(1. + self.sqrootof5*rn + 1.67*rn2)*np.exp(-self.sqrootof5*rn)

    # Derivative with respect to y
    def dkdy(self,l1,l2):
        l = l1[:, None] - l2[None, :]
        rn  = np.abs(l)/self.length
        rn2 = rn**2
        rp  = np.copysign(1,l1[:, None] - l2[None, :])/self.length
        return -self.sigmaSq*rn*rp*1.67*(1. + self.sqrootof5*rn)*np.exp(-self.sqrootof5*rn)


class combinedTrautmanKernel(Kernel):
    # Constructor
    def __init__(self, gamma, sigmaSq, length):
        self.gamma = gamma
        # Covariance magnitude factor
        self.sigmaSq= sigmaSq
        # Characteristic length
        self.length = length
        self.linear = linearKernelTrautman(gamma)
        self.matern = maternKernel(sigmaSq,length)
        self.noise  = 7.50  # Standard deviation for the observation noise
        # Type of kernel
        self.type      = "combinedTrautman"

    # Method to set parameters
    def set_parameters(self,vec):
        self.gamma  = vec[0]
        # Covariance magnitude factor
        self.sigmaSq= vec[1]
        # Characteristic length
        self.length = vec[2]
        self.linear.set_parameters(vec)
        mV = [vec[1], vec[2]]
        self.matern.set_parameters(mV)

    # Overload the operator ()
    def __call__(self,x,y):
        return self.matern(x,y) + self.linear(x,y)

    # Derivative with respect to y
    def dkdy(self,l1,l2):
        dkdy = self.matern.dkdy(l1,l2)
        return dkdy
    # Method to get parameters
    def get_parameters(self):
        parameters = [self.gamma, self.sigmaSq, self.length]
        return parameters

    # Method to print parameters
    def print_parameters(self):
        logging.info("Combined kernel parameters\n gamma ={}\n s ={}, l = {}".format(self.gamma,self.sigmaSq,self.length))



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
        self.linear.set_parameters(vec[2:4])
        self.matern.set_parameters(vec[4:6])

    # Method to set the optimizable parameters
    def set_optimizable_parameters(self,vec):
        # Covariance magnitude factor
        self.sigmaSq       = vec[0]
        # Characteristic length
        self.length        = vec[1]
        self.matern.set_optimizable_parameters(vec)

    # Overload the operator ()
    def __call__(self,l1,l2):
        K = self.matern(l1,l2)
        K+= self.linear(l1,l2)
        return K

    # Derivative with respect to y
    def dkdy(self,l1,l2):
        dkdy = self.matern.dkdy(l1,l2)
        dkdy+= self.linear.dkdy(l1,l2)
        return dkdy

    # Method to get parameters
    def get_parameters(self):
        parameters = [self.meanSlope, self.meanConstant, self.sigmaSlope, self.sigmaConstant, self.sigmaSq, self.length]
        return parameters

    # Method to get the optimizable parameters
    def get_optimizable_parameters(self):
        return [self.sigmaSq, self.length]

    # Method to print parameters
    def print_parameters(self):
        logging.info("combined kernel parameters\n gamma_a = {}\n gamma_0={}\n s ={}, l = {}".format(self.sigmaSlope,self.sigmaConstant,self.sigmaSq,self.length))

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
    def set_parameters(self,vec):
        # Covariance magnitude factor
        self.sigmaSq = vec[0]
        # Characteristic length
        self.length  = vec[1]

    # Overload the operator ()
    def __call__(self,l1,l2):
        l = l1[:, None] - l2[None, :]
        rn  = np.abs(l)/self.length
        return self.sigmaSq*np.exp(-rn)

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
    def set_parameters(self,vec):
        # Covariance magnitude factor
        self.sigmaSq = vec[0]
        # Characteristic length
        self.length  = vec[1]

    # Overload the operator ()
    def __call__(self,l1,l2):
        l = l1[:, None] - l2[None, :]
        rn  = np.abs(l)/self.length
        rn2 = rn**2
        return (self.sigmaSq**2)*np.exp(-0.5*rn2)

    # Derivative with respect to y
    def dkdy(self,x,y):
        return -(self.sigmaSq**2)*(y-x)/(2*self.length**2)*m.exp(-1.*(x-y)**2/(2*self.length**2))
