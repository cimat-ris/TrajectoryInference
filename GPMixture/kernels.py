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

    @abstractmethod
    def __call__(self,x,y): pass

# Linear kernel
class linearKernel(Kernel):
    def __init__(self, gamma_a, gamma_0):
        self.gamma_a = gamma_a
        self.gamma_0 = gamma_0
        self.gamma_non_zero()
        # Type of kernel
        self.type      = "linear"

    def setParameters(self,vec):
        self.gamma_a = vec[0]
        self.gamma_0 = vec[1]
        self.gamma_non_zero()

    def gamma_non_zero(self):
        if(self.gamma_a == 0.):
            self.gamma_a = 0.05
        if(self.gamma_0 == 0.):
            self.gamma_0 = 0.05

    def __call__(self,x,y):
        return (1./self.gamma_a)*x*y + 1./self.gamma_0

    def get_parameters(self):
        parameters = [self.gamma_a, self.gamma_0]
        return parameters


#kernel Trautman
class linearKernelTrautman(Kernel):
    def __init__(self, gamma):
        self.gamma = gamma
        # Type of kernel
        self.type      = "linear"

    def setParameters(self,vec):
        self.gamma = vec[0]

    def __call__(self,x,y):
        return x*y+(1./(self.gamma**2))

# Derived class: the squared exponential kernel
class squaredExponentialKernel(Kernel):
    def __init__(self, fsigma, fl, nsigma):
        self.fsigma = fsigma
        # Characteristic length
        self.length       = length
        self.nsigma = nsigma
        # Type of kernel
        self.type      = "squaredExponential"

    def setParameters(self,vec):
        self.fsigma = vec[0]
        self.fl     = vec[1]

    def __call__(self,x,y):
        return (self.fsigma**2)*m.exp(-1.*(x-y)**2/(2*self.length**2)) + self.nsigma**2*delta(x,y)

# Matern kernel
class maternKernel(Kernel):
    def __init__(self, s, length):
        self.s = s
        # Characteristic length
        self.length       = length
        # Type of kernel
        self.type      = "matern"

    def setParameters(self,vec):
        self.s      = vec[0]
        # Characteristic length
        self.length = vec[1]

    def __call__(self,x,y):
        r = m.fabs(x-y)
        val = self.s*(1. + m.sqrt(5)*r/self.length + 5.*r**2/(3*self.length**2) )*m.exp(-1.*m.sqrt(5)*r/self.length)
        return val

# Matern kernel from Rasmussen
class maternRasmussenKernel(Kernel):
    def __init__(self, s, length):
        # Characteristic length
        self.length = length
        self.s      = s
        # Type of kernel
        self.type      = "maternRasmussen"

    def setParameters(self,vec):
        self.s      = vec[0]
        # Characteristic length
        self.length = vec[1]/self.length

    def __call__(self,x,y):
        rn = m.fabs(x-y)/self.length
        val = self.s*(1. + m.sqrt(3)*rn)*m.exp(-1.*m.sqrt(3)*rn)
        return val

class noiseKernel(Kernel):
    def __init__(self, sigmaNoise):
        self.sigmaNoise = sigmaNoise
        # Type of kernel
        self.type      = "noise"

    def __call__(self,x,y):
        if(x == y):
            return self.sigmaNoise**2.
        else:
            return 0.

# The combination used in the Trautman paper
class combinedTrautmanKernel(Kernel):
    def __init__(self, gamma, s, length, sigmaNoise):
        self.gamma = gamma
        self.s = s
        # Characteristic length
        self.length = length
        self.linear = linearKernel(gamma)
        self.matern = maternKernel(s,length)
        self.noise  = noiseKernel(sigmaNoise)
        # Type of kernel
        self.type      = "combinedTrautman"

    def setParameters(self,vec):
        self.gamma  = vec[0]
        self.s      = vec[1]
        # Characteristic length
        self.length = vec[2]
        self.linear.setParameters(vec)
        mV = [vec[1], vec[2]]
        self.matern.setParameters(mV)

    def __call__(self,x,y):
        return self.matern(x,y) + self.linear(x,y) + self.noise(x,y)

    def get_parameters(self):
        parameters = [self.gamma, self.s, self.length]
        return parameters

    def print_parameters(self):
        print("combined kernel parameters\n gamma =",self.gamma,"\n s =",self.s,", l = ",self.length)


class exponentialKernel(Kernel):
    def __init__(self, s, length):
        self.s = s
        # Characteristic length
        self.length       = length
        # Type of kernel
        self.type      = "exponential"

    def setParameters(self,vec):
        self.s = vec[0]
        # Characteristic length
        self.length = vec[1]

    def __call__(self,x,y):
        return self.s*m.exp(-m.fabs(x-y)/self.length)

class gammaExponentialKernel(Kernel):
    def __init__(self, length, gamma):
        # Characteristic length
        self.length       = length
        self.gamma = gamma
        # Type of kernel
        self.type      = "gammaExponential"

    def setParameters(self,vec):
        # Characteristic length
        self.length= vec[0]
        self.gamma = vec[1]

    def __call__(self,x,y):
        rn = m.fabs(x-y)/self.length
        return m.exp(-rn**self.gamma)

class rationalQuadraticKernel(Kernel):
    def __init__(self, l, alpha, sigmaNoise):
        # Characteristic length
        self.length       = length
        self.alpha = alpha
        self.noise = noiseKernel(sigmaNoise)
        # Type of kernel
        self.type      = "rationalQuadratic"

    def setParameters(self,vec):
        # Characteristic length
        self.length= vec[0]
        self.alpha = vec[1]

    def __call__(self,x,y):
        rn = m.fabs(x-y)/self.alpha*self.length
        return pow(1. + 0.5*(rn**2), -self.alpha) + self.noise(x,y)

class sqrExpKernel(Kernel):
    def __init__(self, l, sigma):
        # Characteristic length
        self.length= length
        self.sigma = sigma

    def setParameters(self,vec):
        # Characteristic length
        self.length= vec[0]
    def __call__(self,x,y):
        return m.exp(-1.*((x-y)**2) / (2*self.l**2))


#kenel combinado: exponencial cuadrado + ruido
class SQKernel(Kernel):
    def __init__(self, length, sigma):
        # Characteristic length
        self.length = length
        self.sigma  = sigma

    def setParameters(self,vec):
        # Characteristic length
        self.length = vec[0]

    def __call__(self,x,y):
        SQker = sqrExpKernel(self.length,self.sigma)
        noise = noiseKernel(self.sigma)
        return SQker(x,y) + noise(x,y)


#kernel combinado: exponencial + ruido
class expKernel(Kernel):
    def __init__(self, s, length, sigma):
        self.exponential = exponentialKernel(s,length)
        self.noise = noiseKernel(sigma)

    def setParameters(self,vec):
        self.exponential.setParameters(vec)

    def __call__(self,x,y):
        return self.exponential(x,y) + self.noise(x,y)

"""************************************"""
#kernel considerando el line prior

#kernel combinado considerando el line prior
class linePriorCombinedKernel(Kernel):
    def __init__(self, gamma_a, gamma_0, s, length, sigmaNoise):
        self.gamma_a = gamma_a
        self.gamma_0 = gamma_0
        self.gamma_non_zero()
        self.s = s
        # Characteristic length
        self.length = length
        self.linear = linearKernel(gamma_a, gamma_0)
        self.matern = maternKernel(s,length)
        self.noise  = noiseKernel(sigmaNoise)

    def setParameters(self,vec):
        self.gamma_a = vec[0]
        self.gamma_0 = vec[1]
        self.gamma_non_zero()
        self.s = vec[2]
        # Characteristic length
        self.length = vec[3]
        self.linear.setParameters(vec)
        mV = [self.s, self.length]
        self.matern.setParameters(mV)

    def gamma_non_zero(self):
        if(self.gamma_a == 0.):
            self.gamma_a = 0.05
        if(self.gamma_0 == 0.):
            self.gamma_0 = 0.05

    def __call__(self,x,y):
        return self.matern(x,y) + self.linear(x,y) + self.noise(x,y)

    def get_parameters(self):
        parameters = [self.gamma_a, self.gamma_0, self.s, self.length]
        return parameters

    def print_parameters(self):
        print("combined kernel parameters\n gamma_a =",self.gamma_a,"\n gamma_0",self.gamma_0,"\n s =",self.s,", l = ",self.length)
