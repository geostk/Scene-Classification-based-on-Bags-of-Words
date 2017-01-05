import numpy as np
from scipy.ndimage import gaussian_filter,gaussian_laplace,correlate1d
import cmath
DEBUG = False
class filter:
    def __init__(self):
        pass


class Gaussian_filter(filter):
    def __init__(self,sigma):
        filter.__init__(self)
        self.sigma = sigma
    def get_filter_response(self,I):
        return gaussian_filter(I,self.sigma)

class LOG_filter(filter):
    def __init__(self,sigma):
        filter.__init__(self)
        self.sigma = sigma
    def get_filter_response(self,I):
        return gaussian_laplace(I,self.sigma)
class Derivative_filterX(filter):
    def __init__(self,sigma):
        filter.__init__(self)
        self.sigma = sigma
    def get_filter_response(self,I):
        return correlate1d(I,[-1,0,1],axis=0)
class Derivative_filterY(filter):
    def __init__(self,sigma):
        filter.__init__(self)
        self.sigma = sigma
    def get_filter_response(self,I):
        return correlate1d(I,[-1,0,1],axis=1)
class filterbank:
    
    def create_filterbanks(self):
            #3 scales
            self.scales = range(1,4);
            if DEBUG:
                print "scales=",self.scales
            #Some arbitrary bandwidths
            self.gaussianSigmas = np.array([1, 2, 4])
            self.logSigmas = np.array([1, 2, 4, 8])
            self.dGaussianSigmas = np.array([2, 4])
            filterBank = [0]*(len(self.scales)*(len(self.gaussianSigmas)+len(self.logSigmas)+2*len(self.dGaussianSigmas)))
            ind = 0
            if DEBUG:
                print "len(filterBank)=",len(filterBank)

            for scale in self.scales:
                scaleMuitiply = (cmath.sqrt(2))**(scale)
                
                for s in self.gaussianSigmas:
                    filterBank[ind] = Gaussian_filter(s*scaleMuitiply)
                    ind=ind+1
                    if DEBUG:
                        print 'ind=',ind

                for s in self.logSigmas:
                    filterBank[ind] = LOG_filter(s*scaleMuitiply)
                    ind=ind+1
                    if DEBUG:
                        print 'ind=',ind
                
                for s in self.dGaussianSigmas:
                    filterBank[ind] = Derivative_filterX(s*scaleMuitiply)
                    ind=ind+1
                    if DEBUG:
                        print 'ind=',ind
                    filterBank[ind] = Derivative_filterY(s*scaleMuitiply)
                    ind=ind+1
                    if DEBUG:
                        print 'ind=',ind

            return filterBank

       

if __name__ == '__main__':
	filterBank = filterbank().create_filterbanks()
	for i in filterBank:
		print type(i), i.sigma