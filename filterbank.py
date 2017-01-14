import numpy as np
from scipy.ndimage import gaussian_filter,gaussian_laplace,correlate1d
import cmath
import matplotlib.pyplot as plt
from PIL import Image
from config import cfg
import pdb
class filter:
    def __init__(self):
        pass


class Gaussian_filter(filter):
    """
    Gaussian filter
    """
    def __init__(self,Sigma):
        filter.__init__(self)
        self.Sigma = Sigma
    def getFilterResponse(self,I):

        return gaussian_filter(I,self.Sigma)

class LOG_filter(filter):
    """
    Laplace filter using gaussian second derivatives
    """
    def __init__(self,Sigma):
        filter.__init__(self)
        self.Sigma = Sigma
    def getFilterResponse(self,I):
        return gaussian_laplace(I,self.Sigma)

class Derivative_filterX(filter):
    """
    d/dx gaussians
    """
    def __init__(self,Sigma):
        filter.__init__(self)
        self.Sigma = Sigma
    def getFilterResponse(self,I):
        return correlate1d(I,[-1,0,1],axis=0)
class Derivative_filterY(filter):
    """
    d/dy gaussians
    """
    def __init__(self,Sigma):
        filter.__init__(self)
        self.Sigma = Sigma
    def getFilterResponse(self,I):
        return correlate1d(I,[-1,0,1],axis=1)

    
def createFilterbanks():
    """
    Return a list of filter objects as filterbank
    """
    #3 scales
    Scales = cfg.SCALES_FOR_CREATINMG_FILTERBANK
    #Some bandwidths
    GaussianSigmas = cfg.GAUSSIAN_SIGMAS
    LogSigmas = cfg.LOG_SIGMAS
    DGaussianSigmas =cfg.D_GAUSSIAN_SIGMAS
    FilterBank = [0]*(len(Scales)*(len(GaussianSigmas)+len(LogSigmas)+2*len(DGaussianSigmas)))
    ind = 0

    for scale in Scales:
        ScaleMuitiply = (np.sqrt(2))**(scale)
        for s in GaussianSigmas:
            FilterBank[ind] = Gaussian_filter(s*ScaleMuitiply)
            ind=ind+1
            
        for s in LogSigmas:
            FilterBank[ind] = LOG_filter(s*ScaleMuitiply)
            ind=ind+1
                           
        for s in DGaussianSigmas:
            FilterBank[ind] = Derivative_filterX(s*ScaleMuitiply)
            ind=ind+1
            
            FilterBank[ind] = Derivative_filterY(s*ScaleMuitiply)
            ind=ind+1
           
    return FilterBank



if __name__ == '__main__':
    FilterBank = createFilterbanks()
    
