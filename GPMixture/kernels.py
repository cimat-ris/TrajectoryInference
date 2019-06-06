from abc import ABCMeta, abstractmethod
import math as m

#Delta de Kronecker
def delta(x,y):
    if x == y:
        return 1
    else:
        return 0

# Abstract class for handling kernels
class Kernel:
    __metaclass__ = ABCMeta

    @abstractmethod
    def k(self,x,y): pass

# Derived class: the squared exponential kernel        
class squaredExponentialKernel(Kernel):
    def __init__(self, fsigma, fl, nsigma):
        self.fsigma = fsigma    
        self.fl     = fl    
        self.nsigma = nsigma

    def setParameters(self,vec):
        self.fsigma = vec[0]    
        self.fl     = vec[1] 

    def k(self,x,y):
        return (self.fsigma**2)*m.exp(-1.*(x-y)**2/(2*self.fl**2)) + self.nsigma**2*delta(x,y)

#kernel Trautman
class linearKernel(Kernel):
    def __init__(self, gamma):
        self.gamma = gamma
        
    def setParameters(self,vec):
        self.gamma = vec[0]    

    def k(self,x,y):
        return x*y+(1./(self.gamma**2))
        
#kernel Trautman
class maternKernel(Kernel):
    def __init__(self, s, l):
        self.s = s
        self.l = l
        
    def setParameters(self,vec):
        self.s = vec[0]    
        self.l = vec[1] 

    def k(self,x,y):
        r = m.fabs(x-y)
        val = self.s*(1. + m.sqrt(5)*r/self.l + 5.*r**2/(3*self.l**2) )*m.exp(-1.*m.sqrt(5)*r/self.l)
        return val
        
class noiseKernel(Kernel):
    def __init__(self, sigma):
        self.sigma = sigma
   
    def k(self,x,y):
        if(x == y):
            return self.sigma**2.
        else:
            return 0.
            
#kernel del Trautman
#kernel combinado: matern + lineal + ruido
class combinedKernel(Kernel):
    def __init__(self, gamma, s, l, sigma):
        self.gamma = gamma
        self.s = s
        self.l = l
        self.linear = linearKernel(gamma)
        self.matern = maternKernel(s,l)
        self.noise = noiseKernel(sigma)
        
    def setParameters(self,vec):
        self.gamma = vec[0]
        self.s = vec[1]
        self.l = vec[2]
        self.linear.setParameters(vec)
        mV = [vec[1], vec[2]]
        self.matern.setParameters(mV)

    def k(self,x,y):
        return self.matern.k(x,y) + self.linear.k(x,y) + self.noise.k(x,y) 
        
    def get_parameters(self):
        parameters = [self.gamma, self.s, self.l]
        return parameters
        
    def print_parameters(self):
        print("combined kernel parameters\n gamma =",self.gamma,"\n s =",self.s,", l = ",self.l)
        
#kernel modificado para la optimizacion, quitamos el parametro gamma
#kernel combinado:  lineal + matern + ruido
class combinedKernelModified(Kernel):
    def __init__(self, s, l, sigma):
        self.s = s
        self.l = l
        self.matern = maternKernel(s,l)
        self.noise = noiseKernel(sigma)
        
    def setParameters(self,vec):
        self.s = vec[0]
        self.l = vec[1]
        mV = [vec[0], vec[1]]
        self.matern.setParameters(mV)

    def k(self,x,y):
        return x*y + self.matern.k(x,y) + self.noise.k(x,y)
        
    def get_parameters(self):
        parameters = [self.s, self.l]
        return parameters
        
    def print_parameters(self):
        print("combined kernel parameters\n","s =",self.s,", l = ",self.l)
        
class exponentialKernel(Kernel):
    def __init__(self, s, l): 
        self.s = s
        self.l = l    

    def setParameters(self,vec):
        self.s = vec[0] 
        self.l = vec[1] 

    def k(self,x,y):
        return self.s*m.exp(-m.fabs(x-y)/self.l)
        
class gammaExponentialKernel(Kernel):
    def __init__(self, l, gamma):   
        self.l     = l    
        self.gamma = gamma

    def setParameters(self,vec):
        self.l     = vec[0]
        self.gamma = vec[1]

    def k(self,x,y):
        r = m.fabs(x-y)
        return m.exp(- (r/self.l)**self.gamma)
        
class rationalQuadraticKernel(Kernel):
    def __init__(self, l, alpha, sigma):   
        self.l     = l
        self.alpha = alpha
        self.noise = noiseKernel(sigma)

    def setParameters(self,vec):
        self.l = vec[0] 
        self.alpha = vec[1] 

    def k(self,x,y):
        r = m.fabs(x-y)
        return pow(1. + (r**2)/(2*self.alpha*self.l**2), -self.alpha) + self.noise.k(x,y)    
        
class sqrExpKernel(Kernel):
    def __init__(self, l, sigma):   
        self.l     = l
        self.sigma = sigma

    def setParameters(self,vec):
        self.l = vec[0] 

    def k(self,x,y):
        return m.exp(-1.*((x-y)**2) / (2*self.l**2))
        
#kernel matern del Rasmusen
class matKernel(Kernel):
    def __init__(self, s, l):   
        self.l = l
        self.s = s

    def setParameters(self,vec):
        self.s = vec[0] 
        self.l = vec[1] 

    def k(self,x,y):
        r = m.fabs(x-y)
        #return self.s*(1. + m.sqrt(3)*r/self.l)*m.exp(-1.*m.sqrt(3)*r/self.l)
        val = self.s*(1. + m.sqrt(3)*r/self.l)*m.exp(-1.*m.sqrt(3)*r/self.l)
        return val
        
        
#kenel combinado: exponencial cuadrado + ruido
class SQKernel(Kernel):
    def __init__(self, l, sigma):   
        self.l     = l
        self.sigma = sigma

    def setParameters(self,vec):
        self.l = vec[0] 

    def k(self,x,y):
        SQker = sqrExpKernel(self.l,self.sigma)
        noise = noiseKernel(self.sigma)
        return SQker.k(x,y) + noise.k(x,y)
        
        
#kernel combinado: exponencial + ruido
class expKernel(Kernel):
    def __init__(self, s, l, sigma):
        self.exponential = exponentialKernel(s,l)
        self.noise = noiseKernel(sigma)
        
    def setParameters(self,vec):
        self.exponential.setParameters(vec)

    def k(self,x,y):
        return self.exponential.k(x,y) + self.noise.k(x,y)
        
"""************************************"""
#kernel considerando el line prior

#kernel Trautman
class linePriorLinearKernel(Kernel):
    def __init__(self, gamma_a, gamma_0):
        self.gamma_a = gamma_a
        self.gamma_0 = gamma_0
        self.gamma_non_zero()
        
    def setParameters(self,vec):
        self.gamma_a = vec[0]   
        self.gamma_0 = vec[1] 
        self.gamma_non_zero()

    def gamma_non_zero(self):
        if(self.gamma_a == 0.):
            self.gamma_a = 0.05
        if(self.gamma_0 == 0.):
            self.gamma_0 = 0.05
            
    def k(self,x,y):
        return (1./self.gamma_a)*x*y + 1./self.gamma_0
        
    def get_parameters(self):
        parameters = [self.gamma_a, self.gamma_0]
        return parameters
        
#kernel combinado considerando el line prior
class linePriorCombinedKernel(Kernel):
    def __init__(self, gamma_a, gamma_0, s, l, sigma):
        self.gamma_a = gamma_a
        self.gamma_0 = gamma_0
        self.gamma_non_zero()
        self.s = s
        self.l = l
        self.linear = linePriorLinearKernel(gamma_a, gamma_0)
        self.matern = maternKernel(s,l)
        self.noise = noiseKernel(sigma)
        
    def setParameters(self,vec):
        self.gamma_a = vec[0]
        self.gamma_0 = vec[1]
        self.gamma_non_zero()
        self.s = vec[2]
        self.l = vec[3]
        self.linear.setParameters(vec)
        mV = [self.s, self.l]
        self.matern.setParameters(mV)

    def gamma_non_zero(self):
        if(self.gamma_a == 0.):
            self.gamma_a = 0.05
        if(self.gamma_0 == 0.):
            self.gamma_0 = 0.05

    def k(self,x,y):
        return self.matern.k(x,y) + self.linear.k(x,y) + self.noise.k(x,y) 
    
    def get_parameters(self):
        parameters = [self.gamma_a, self.gamma_0, self.s, self.l]
        return parameters
        
    def print_parameters(self):
        print("combined kernel parameters\n gamma_a =",self.gamma_a,"\n gamma_0",self.gamma_0,"\n s =",self.s,", l = ",self.l)
      