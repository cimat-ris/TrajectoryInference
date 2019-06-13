from abc import ABCMeta, abstractmethod
import math as m

# Kronecker delta
def delta(x,y):
    if x == y:
        return 1
    else:
        return 0

# Abstract class for handling kernels
class Kernel:
    __metaclass__ = ABCMeta

    def __init__(self):
        # Type of kernel
        self.type      = "generic"

    # Overload the operator ()
    @abstractmethod
    def __call__(self,x,y): pass

# Linear kernel
class linearKernel(Kernel):
    def __init__(self, sigma_a, sigma_c):
        # Variance on the slope
        self.sigma_a   = sigma_a
        self.sigmaSq_a = sigma_a**2
        # Variance on the constant
        self.sigma_c   = sigma_c
        self.sigmaSq_c = sigma_c**2
        # Type of kernel
        self.type      = "linear"

    def setParameters(self,vec):
        # Variance on the slope
        self.sigma_a   = vec[0]
        self.sigmaSq_a = vec[0]**2
        # Variance on the constant
        self.sigma_c   = vec[1]
        self.sigmaSq_c = vec[1]**2

    def get_parameters(self):
        parameters = [self.sigma_a, self.sigma_c]
        return parameters

    # Overload the operator ()
    def __call__(self,x,y):
        return self.sigmaSq_a*x*y+self.sigmaSq_c


#kernel Trautman
class linearKernelTrautman(Kernel):
    def __init__(self, gamma):
        self.gamma = gamma
        # Type of kernel
        self.type      = "linear"

    def setParameters(self,vec):
        self.gamma = vec[0]

    # Overload the operator ()
    def __call__(self,x,y):
        return x*y+(1./(self.gamma**2))

# Derived class: the squared exponential kernel
class squaredExponentialKernel(Kernel):
    def __init__(self, sigmaSq, length):
        # Covariance magnitude factor
        self.sigmaSq      = sigmaSq
        # Characteristic length
        self.length       = length
        # Type of kernel
        self.type         = "squaredExponential"

    def setParameters(self,vec):
        # Covariance magnitude factor
        self.sigmaSq = vec[0]
        # Characteristic length
        self.length  = vec[1]

    # Overload the operator ()
    def __call__(self,x,y):
        return (self.sigmaSq**2)*m.exp(-1.*(x-y)**2/(2*self.length**2))

# Matern kernel
class maternKernel(Kernel):
    def __init__(self, sigmaSq, length):
        # Covariance magnitude factor
        self.sigmaSq      = sigmaSq
        # Characteristic length
        self.length       = length
        # Type of kernel
        self.type         = "matern"
        self.sqrootof5    = m.sqrt(5)
    def setParameters(self,vec):
        # Covariance magnitude factor
        self.sigmaSq= vec[0]
        # Characteristic length
        self.length = vec[1]

    # Overload the operator ()
    def __call__(self,x,y):
        rn  = m.fabs((x-y)/self.length)
        # To avoid overflow
        if rn>20.0:
            val = 0.0
        else:
            rn2 = rn**2
            val = self.sigmaSq*(1. + self.sqrootof5*rn + 1.67*rn2)*m.exp(-self.sqrootof5*rn)
        return val

# Matern kernel from Rasmussen
class maternRasmussenKernel(Kernel):
    def __init__(self, sigmaSq, length):
        # Characteristic length
        self.length = length
        # Covariance magnitude factor
        self.sigmaSq= sigmaSq
        # Type of kernel
        self.type   = "maternRasmussen"

    def setParameters(self,vec):
        self.sigmaSq      = vec[0]
        # Characteristic length
        self.length       = vec[1]

    # Overload the operator ()
    def __call__(self,x,y):
        rn = m.fabs(x-y)/self.length
        val = self.sigmaSq*(1. + m.sqrt(3)*rn)*m.exp(-1.*m.sqrt(3)*rn)
        return val

class noiseKernel(Kernel):
    def __init__(self, sigmaNoise):
        # Standard deviation of the noise
        self.sigmaNoise = sigmaNoise
        # Type of kernel
        self.type       = "noise"

    # Overload the operator ()
    def __call__(self,x,y):
        if(x == y):
            return self.sigmaNoise**2.
        else:
            return 0.

# The combination used in the Trautman paper
class combinedTrautmanKernel(Kernel):
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

    def get_parameters(self):
        parameters = [self.gamma, self.sigmaSq, self.length]
        return parameters

    def print_parameters(self):
        print("combined kernel parameters\n gamma =",self.gamma,"\n s =",self.s,", l = ",self.length)

# Exponential kernel
class exponentialKernel(Kernel):
    def __init__(self, sigmaSq, length):
        # Covariance magnitude factor
        self.sigmaSq   = sigmaSq
        # Characteristic length
        self.length    = length
        # Type of kernel
        self.type      = "exponential"

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
    def __init__(self, sigmaSq, length, gamma):
        # Covariance magnitude factor
        self.sigmaSq = sigmaSq
        # Characteristic length
        self.length       = length
        # Gamma exponent
        self.gamma = gamma
        # Type of kernel
        self.type      = "gammaExponential"

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
    def __init__(self, sigmaSq, length, alpha, sigmaNoise):
        # Covariance magnitude factor
        self.sigmaSq = sigmaSq
        # Characteristic length
        self.length  = length
        self.alpha   = alpha
        self.noise   = noiseKernel(sigmaNoise)
        # Type of kernel
        self.type    = "rationalQuadratic"

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
    def __init__(self, sigmaSq, length, sigmaNoise):
        # Covariance magnitude factor
        self.sigmaSq      = sigmaSq
        # Characteristic length
        self.length      = length
        # Standard deviation of the noise
        self.sigmaNoise  = sigmaNoise
        # Type of kernel
        self.type    = "squaredExponentialAndNoise"

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
    def __init__(self, sigmaSq, length, sigmaNoise):
        self.exponential = exponentialKernel(sigmaSq,length)
        self.noise       = noiseKernel(sigmaNoise)
        # Type of kernel
        self.type    = "exponentialAndNoise"

    def setParameters(self,vec):
        self.exponential.setParameters(vec)

    def __call__(self,x,y):
        return self.exponential(x,y) + self.noise(x,y)

# A combined kernel that considers a prior on the line parameters
class linePriorCombinedKernel(Kernel):
    def __init__(self, sigmaSlope, sigmaConstant, sigmaSq, length, sigmaNoise):
        self.sigmaSlope    = sigmaSlope
        self.sigmaConstant = sigmaConstant
        # Covariance magnitude factor
        self.sigmaSq= sigmaSq
        # Characteristic length
        self.length = length
        self.linear = linearKernel(sigmaSlope, sigmaConstant)
        self.matern = maternKernel(sigmaSq,length)
        self.noise  = noiseKernel(sigmaNoise)
        # Type of kernel
        self.type    = "linePriorCombined"

    def set_parameters(self,vec):
        self.sigmaSlope    = vec[0]
        self.sigmaConstant = vec[1]
        # Covariance magnitude factor
        self.sigmaSq       = vec[2]
        # Characteristic length
        self.length        = vec[3]
        self.linear.setParameters(vec)
        mV = [self.sigmaSq, self.length]
        self.matern.setParameters(mV)

    def set_optimizable_parameters(self,vec):
        # Covariance magnitude factor
        self.sigmaSq       = vec[0]
        # Characteristic length
        self.length        = vec[1]
        self.matern.setParameters(vec)

    def __call__(self,x,y):
        return self.matern(x,y) + self.linear(x,y) + self.noise(x,y)

    def get_parameters(self):
        parameters = [self.sigmaSlope, self.sigmaConstant, self.sigmaSq, self.length]
        return parameters

    def get_optimizable_parameters(self):
        parameters = [self.sigmaSq, self.length]
        return parameters

    def print_parameters(self):
        print("combined kernel parameters\n gamma_a =",self.sigmaSlope,"\n gamma_0",self.sigmaConstant,"\n s =",self.sigmaSq,", l = ",self.length)
