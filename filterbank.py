import numpy as np
from scipy.ndimage import gaussian_filter,gaussian_laplace,correlate1d
import cmath
DEBUG = False
class filter:
    def __init__(self):
        pass


class Gaussian_filter(filter):
    def __init__(self,Sigma):
        filter.__init__(self)
        self.Sigma = Sigma
    def get_filter_response(self,I):
        return gaussian_filter(I,self.Sigma)

class LOG_filter(filter):
    def __init__(self,Sigma):
        filter.__init__(self)
        self.Sigma = Sigma
    def get_filter_response(self,I):
        return gaussian_laplace(I,self.Sigma)
class Derivative_filterX(filter):
    def __init__(self,Sigma):
        filter.__init__(self)
        self.Sigma = Sigma
    def get_filter_response(self,I):
        return correlate1d(I,[-1,0,1],axis=0)
class Derivative_filterY(filter):
    def __init__(self,Sigma):
        filter.__init__(self)
        self.Sigma = Sigma
    def get_filter_response(self,I):
        return correlate1d(I,[-1,0,1],axis=1)

    
def createFilterbanks():
        #3 scales
    Scales = range(1,4);
    if DEBUG:
        print "scales=",Scales
    #Some arbitrary bandwidths
    GaussianSigmas = np.array([1, 2, 4])
    LogSigmas = np.array([1, 2, 4, 8])
    DGaussianSigmas = np.array([2, 4])
    FilterBank = [0]*(len(Scales)*(len(GaussianSigmas)+len(LogSigmas)+2*len(DGaussianSigmas)))
    ind = 0
    if DEBUG:
        print "Number of filters = ",len(FilterBank)

    for scale in Scales:
        ScaleMuitiply = (cmath.sqrt(2))**(scale)
        for s in GaussianSigmas:
            FilterBank[ind] = Gaussian_filter(s*ScaleMuitiply)
            ind=ind+1
            if DEBUG:
                print 'ind=',ind

        for s in LogSigmas:
            FilterBank[ind] = LOG_filter(s*ScaleMuitiply)
            ind=ind+1
            if DEBUG:
                print 'ind=',ind
                
        for s in DGaussianSigmas:
            FilterBank[ind] = Derivative_filterX(s*ScaleMuitiply)
            ind=ind+1
            if DEBUG:
                print 'ind=',ind
            FilterBank[ind] = Derivative_filterY(s*ScaleMuitiply)
            ind=ind+1
            if DEBUG:
                print 'ind=',ind

    return FilterBank

       

if __name__ == '__main__':
	FilterBank = createFilterbanks()
	for i in FilterBank:
		print i.Sigma