from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
import scipy
import scipy.io as sio
import pdb
from RGB2Lab import rgb2Lab
import random
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import os
import cPickle
from scipy.cluster.vq import kmeans
from multiprocessing import Pool
from filterbank import createFilterbanks
from config import cfg

def MPWorkerForGetFilterResponses(x):
    """
    Worker for multiprocess to extract filterResponses from an image
    """
    ImPath,FilterBank,alpha = x
    print "getting filter responses from ",ImPath
    I = Image.open(ImPath)
    FilterResponse = extractFilterResponses(I,FilterBank)
    RandomPixelIndx=random.sample(range(FilterResponse.shape[0]),alpha)
    return FilterResponse[RandomPixelIndx,:]

def extractFilterResponses(Im,FilterBank):
    """
    input parameters:
        Im : the input image (type:PIL.JpegImagePlugin.JpegImageFile),
            opened by Image.open()
        FilterBank : a list of filters

    return: filterResponses:  A numpy.array(N-by-3*M).
            N is number of pixls, M is number of filters, 3 channels.

    """
    
    N = (Im.size)[0] * (Im.size)[1] # N is the number of pixels of the image I
    M = len(FilterBank) #Number of filters
    FilterResponses = np.zeros((N,M*3))
    (L,a,b) = rgb2Lab(Im)
    for i, f in enumerate(FilterBank):
        FilterResponses[:,3*i]=np.reshape((f.getFilterResponse(L)),(N,))
        FilterResponses[:,3*i+1]=np.reshape((f.getFilterResponse(a)),(N,))
        FilterResponses[:,3*i+2]=np.reshape((f.getFilterResponse(b)),(N,))
    return FilterResponses


def getFilterBankAndDictionary(imPaths):
    """
    Description:

    Creates filterbanks, uses the filters to get filter responses from each 
    given image then randomly choose some pixels from these image to compute
    dictionary with Kmeans.
    Use multiprocess to accelerate the process of extracting filter responses 

    input parameters:
        imPaths : A list of string represents the path of training images

    return:
        (FilterBank,Dictionary)
        A tuple contains filterbank and dictionary 

    """
    
    FilterBank = createFilterbanks()

    alpha = cfg.ALPHA# choose alpha pixels from an image to compute dictionary
    K = cfg.K_FOR_KMEANS # K for Kmeans
    T = len(imPaths)
    
    ''' ############Sequence solution########################
    N= 3*len(FilterBank)
    
    filterResponses = np.zeros((alpha*T,N))
    for i,imPath in enumerate(imPaths):
        print 'i=',i,'imPath=',imPath[0]
        I = Image.open(imPath[0])
        
        filterResponse = extractFilterResponses(I,FilterBank)

        random_pixels=random.sample(range(filterResponse.shape[0]),alpha)
        filterResponses[i:i+alpha,:]=filterResponse[random_pixels,:]
    '''
        ##################################################
    p = Pool(None)
    x = [ (i,FilterBank,alpha) for i in imPaths]
    FilterResponses = p.map(MPWorkerForGetFilterResponses, x)
    p.close()
    p.join()
    FilterResponses = np.array(FilterResponses).reshape(T*alpha,-1)
############################################################
    
    print 'Computing Kmeans\n'
    Dictionary = kmeans(FilterResponses, K,iter =cfg.K_FOR_KMEANS)#iter=200
    print 'Done\n'
    return (FilterBank,Dictionary)

def computeDictionary(TrainImagePaths,ImagesDir):
    """
    Description:

    Construct the path for each training image, get filterbank and compute 
    dictionary.
    Save dictionary and filterbank for future use.

    input parameters:
        TrainImagePaths: all paths of training images
        ImagesDir: path to the image directory

    """
    #pdb.set_trace()
    NewTrainImagePaths=[]
    for i, path in enumerate(TrainImagePaths):        
        NewTrainImagePaths.append((os.path.join(ImagesDir,path[0][0])))
        print i, TrainImagePaths[i]

    (FilterBank,Dictionary) = getFilterBankAndDictionary(NewTrainImagePaths)
    cPickle.dump(Dictionary,open('dictionary.pkl', 'wb'))
    cPickle.dump(FilterBank,open('filterbank.pkl', 'wb'))
    print Dictionary,
    return Dictionary

def getVisualWords(I, FilterBank, dictionary):
    '''
    Descriptipon:
    Map each pixel in the image to its closest word in the dictionary

    '''

    FilterResponses = extractFilterResponses(I,FilterBank)
    """
    Compute the distance between filterResponses and dictionary words
    D(i,j) represent the distance between i th filterResponses and j th word 
    in dictionary
    """
    D = cdist(FilterResponses,dictionary[0],'euclidean')
    WordMap = (np.argmin(D,axis=1)).reshape((I.size[1],I.size[0]))
    return WordMap



if __name__== "__main__":
    images_dir = os.path.join('images')
    assert os.path.exists(images_dir), "Paht{} not exists".format(images_dir)
    train_test_mat_file = sio.loadmat('traintest.mat')
    train_image_paths = train_test_mat_file['trainImagePaths']
    #computeDictionary(train_image_paths,images_dir)
    I= Image.open('sun_aaegrbxogokacwmz.jpg')
    pdb.set_trace()
    WordMap=getVisualWords(I,createFilterbanks(), cPickle.load(open('dictionary.pkl', 'rb')))
    plt.imshow(WordMap)
    plt.show()
    '''
    f1=filterBank.Gaussian_filter(1)
    I1=f1.get_filter_response(im)

    f2=filterBank.LOG_filter(1)
    I2=f2.get_filter_response(im)
    plt.figure(1)
    plt.imshow(im,cmap ='gray')
    plt.figure(2)
    plt.imshow(im,cmap =  plt.cm.gray)
    plt.figure(3)
    plt.imshow(im,cmap = plt.cm.gray_r)
    plt.show() 
    '''
