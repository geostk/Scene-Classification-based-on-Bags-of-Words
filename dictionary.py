from PIL import Image
import filterBank
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
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
def extractFilterResponses(I,filterBank):
    """
    I is numpy.array

    return: filterResponses(N-by-3*M)
            N is number of pixls, M is number of filters
    """
    #I.dtype = I.astype(float)
    N = (I.size)[0] * (I.size)[1] # number of pixl
    M = len(filterBank)
    filterResponses = np.zeros((N,M*3))
    (L,a,b) = rgb2Lab(I)
    for i, f in enumerate(filterBank):
        filterResponses[:,3*i]=np.reshape((f.get_filter_response(L)),(N,))
        filterResponses[:,3*i+1]=np.reshape((f.get_filter_response(a)),(N,))
        filterResponses[:,3*i+2]=np.reshape((f.get_filter_response(b)),(N,))
    return filterResponses


def getFilterBankAndDictionary(imPaths):
    
    filter_Bank = filterBank.filterbank().create_filterbanks();
    alpha = 100;
    K = 200;
    N= 3*len(filter_Bank)
    T = len(imPaths)
    filterResponses = np.zeros((alpha*T,N))
    for i,imPath in enumerate(imPaths):
        print 'i=',i,'imPath=',imPath[0]
        I = Image.open(imPath[0])
        
        filterResponse = extractFilterResponses(I,filter_Bank)

        random_pixels=random.sample(range(filterResponse.shape[0]),alpha)
        filterResponses[i:i+alpha,:]=filterResponse[random_pixels,:]
    dictionary = kmeans(filterResponses, K,iter=200)
    return dictionary

def computeDictionary(train_image_paths,images_dir):

    for i, path in enumerate(train_image_paths):
        
        train_image_paths[i] = np.array(os.path.join(images_dir,path[0][0]))
        print i, train_image_paths[i][0]

    dictionary = getFilterBankAndDictionary(train_image_paths.tolist())
    cPickle.dump(dictionary,open('dictionary.pkl', 'wb'))
    print dictionary,
    return dictionary

def getVisualWords(I, filter_bank, dictionary):
    filterResponses = extractFilterResponses(I,filter_bank)
    pdb.set_trace()
    D = cdist(filterResponses,dictionary[0],'euclidean')
    wordMat = (np.argmin(D,axis=1)).reshape((I.size[0],I.size[1]))
    return wordMat



if __name__== "__main__":
    images_dir = os.path.join('images')
    assert os.path.exists(images_dir), "Paht{} not exists".format(images_dir)
    train_test_mat_file = sio.loadmat('traintest.mat')
    train_image_paths = train_test_mat_file['trainImagePaths']
    computeDictionary(train_image_paths,images_dir)
    I= Image.open('sun_aaegrbxogokacwmz.jpg')
    getVisualWords(I, filterBank.filterbank().create_filterbanks(), cPickle.load(open('dictionary.pkl', 'rb')))
    print "AAAAAAAA",
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
