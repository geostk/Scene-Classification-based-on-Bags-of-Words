import numpy as np
import pdb
import cPickle
import dictionary 
import filterBank
from PIL import Image
import matplotlib.pyplot as plt
import scipy.io as sio
from dictionary import computeDictionary,getVisualWords
import os
from multiprocessing import Pool
from scipy import stats
def getImageFeatures(wordMap,dictionarySize):
    
    hist = np.zeros((dictionarySize,))
    numOfPixel = wordMap.size
    wordMap=np.reshape(wordMap,(numOfPixel,))
    for i in xrange(numOfPixel):
        hist[wordMap[i]]=hist[wordMap[i]]+1
   

    return hist/numOfPixel
def getImageFeaturesSPM(LayerNum, WordMap, DictionarySize):
 
    L=LayerNum-1
    (H,W) = WordMap.shape
    CellHeight=H/(2**L)
    CellWidth=W/(2**L)
    LayerHist=np.zeros(((4**L)*DictionarySize,))
    SavedHist=np.zeros((4**L,DictionarySize)) # Save finer layer histograms for computing the next layer
    NewHist=np.zeros((4**L,DictionarySize)) # Save layer histograms for updating saved_hist
    for i in xrange(2**L):   #1:2^L
        for j in xrange(2**L):#j=1:2^L
            StartColum=i*CellWidth
            StartRow=j*CellHeight
            CurrentHist = getImageFeatures(WordMap[StartRow:StartRow+CellHeight,StartColum:StartColum+CellWidth],DictionarySize)
            LayerHist[i*(2**L)+j*DictionarySize:i*(2**L)+j*DictionarySize+DictionarySize]=CurrentHist
            SavedHist[i*(2**L)+j,:]=CurrentHist
            
# normalize
    LayerHist = LayerHist / (4**L)
    if(L>1):
        weight=2**(-1)
    else:
        weight=2**(-L)
    
    Hist=LayerHist*weight



    for l in reversed(range(L)):# L-1:-1:0
       
        LayerHist=np.zeros(((4**L)*DictionarySize))
        
        for i in xrange(1,2**l+1):#1:2^l
            for j in xrange(1,2**l+1):#j=1:2^l
                NewHist[(i-1)*(2**l)+j-1,:] = \
                    SavedHist[0+(i-1)*(2**(l+2))+(j-1)*2,:]+ \
                    SavedHist[1+(i-1)*(2**(l+2))+(j-1)*2,:]+ \
                    SavedHist[0+(i-1)*(2**(l+2))+(j-1)*2+2^(l+1),:]+ \
                    SavedHist[1+(i-1)*(2**(l+2))+(j-1)*2+2^(l+1),:]
                
                LayerHist[i*(2**L)+j*DictionarySize:i*(2**L)+j*DictionarySize+DictionarySize]= NewHist[(i-1)*(2**l)+j-1,:]/4
            
       
        SavedHist=NewHist/4;
        if(l>1):
            weight=2**(l-L-1);
        else:
            weight=2**(-L);
        
        #normalize
        LayerHist =LayerHist / (4**l);
        Hist=np.hstack((LayerHist*weight,Hist));
    #pdb.set_trace()    
    return Hist

def distanceToSet(WordHist, Histograms):
     WordHist = WordHist.reshape((-1,1))
     WordHist =np.tile(WordHist,(1,Histograms.shape[1]))

     HistInter = np.sum(np.minimum(WordHist,Histograms),axis=0)
     return HistInter

def trainSystem():
    TrainTestMatFile = sio.loadmat('traintest.mat')
    ImageDir = 'images'
    TargetDir = './wordmap'
    TrainImagePaths = TrainTestMatFile['trainImagePaths']
    Classnames = TrainTestMatFile['classnames']
    #pdb.set_trace()
    print "Computing dictionary ... ",
    computeDictionary(TrainImagePaths,ImageDir)
    print "Computing dictionary done... ",
    Dictionary = cPickle.load(open('dictionary.pkl', 'rb'))
    FilterBank = cPickle.load(open('filterbank.pkl', 'rb'))
    print 'Computing visual words ... ',
    batchToVisualWords(TrainImagePaths,Classnames,FilterBank,Dictionary,ImageDir,TargetDir,10)
    print 'Done',
    TrainHistograms=createHistograms(len(Dictionary[0]),TrainImagePaths,TargetDir)
    cPickle.dump(TrainHistograms,open('TrainHistograms.pkl', 'wb'))
    #pdb.set_trace()




def batch(x):
    ImagePath,FilterBank,Dictionary,ImageDir,TargetDir = x

    print 'openning image:',ImagePath[0][0]
    I = Image.open(os.path.join(ImageDir,ImagePath[0][0]))
    #I = Image.open(ImagePath[0][0])
    print 'Converting to visual words {0}\n'.format(ImagePath[0][0])
    WordRepresentation = getVisualWords(I, FilterBank, Dictionary)
    OutPutPath = os.path.join(TargetDir,ImagePath[0][0]+'.pkl')
    cPickle.dump(WordRepresentation,open(OutPutPath, 'wb'))

def batchToVisualWords(TrainImagePaths,Classnames,FilterBank,Dictionary,ImageDir,TargetDir,NumOfCores):
    
    if not os.path.exists(TargetDir):
        os.mkdir(TargetDir)
    for c in xrange(len(Classnames)):
        temppath=os.path.join(TargetDir,Classnames[c,0][0])
        if not os.path.exists(temppath):
            os.mkdir(os.path.join(TargetDir,Classnames[c,0][0]))

    x = [(a,FilterBank,Dictionary,ImageDir,TargetDir) for a in TrainImagePaths]
    #pdb.set_trace()
    p = Pool(None)
    p.map(batch, x)
    p.close()
    p.join()

def createHistograms(DictionarySize,TrainImagePaths,TargetDir):
    LayerNum=3
    ImageDir='./images'
    OutPutHistograms = np.zeros((DictionarySize*(48),len(TrainImagePaths)))
    for i in xrange(len(TrainImagePaths)):
        print 'createHistograms',i
        WordMap = cPickle.load(open(os.path.join(TargetDir,TrainImagePaths[i][0][0]+'.pkl')))
        OutPutHistograms[:,i] = getImageFeaturesSPM(LayerNum,WordMap,DictionarySize)
    #pdb.set_trace() 
    return OutPutHistograms


def evaluateRecognitionSystem():
    ImageDir='./images'
    TrainTestMatFile = sio.loadmat('traintest.mat')
    TestImagePaths = TrainTestMatFile['testImagePaths']
    TrainImageLabels = TrainTestMatFile['trainImageLabels']
    TestImageLabels  = TrainTestMatFile['testImageLabels']
    Dictionary = cPickle.load(open('dictionary.pkl', 'rb'))
    FilterBank = cPickle.load(open('filterbank.pkl', 'rb'))
    TrainHistograms = cPickle.load(open('TrainHistograms.pkl', 'rb'))
    DictionarySize=len(Dictionary[0])
    k=2
    ConfusionMatrix=np.zeros((9,9))
    LayerNum=3
    NumOfTestImages=len(TestImagePaths)
    for i in xrange(NumOfTestImages):
        print i,'/',NumOfTestImages
        I = Image.open(os.path.join(ImageDir,TestImagePaths[i][0][0]))
        WordMap=getVisualWords(I,FilterBank,Dictionary)
        WordHist=getImageFeaturesSPM(LayerNum,WordMap,DictionarySize)
        predictedLabel = knnClassify(WordHist,TrainHistograms,TrainImageLabels,k)
        
        ConfusionMatrix[TestImageLabels[i][0]-1,predictedLabel-1]=ConfusionMatrix[TestImageLabels[i][0]-1,predictedLabel-1]+1;
    Accuracy = np.trace(ConfusionMatrix)/np.sum(ConfusionMatrix)
    print 'Accuracy=',Accuracy
def knnClassify(WordHist,TrainHistograms,TrainImageLabels,k):
    Distances = distanceToSet(WordHist, TrainHistograms)
    I = np.argsort(Distances)[::-1]
    PredictedLabel= stats.mode(TrainImageLabels[I[1:k]])
    return PredictedLabel[0][0][0]

if __name__ == '__main__':
    #Dictionary = cPickle.load(open('dictionary.pkl', 'rb'))
    #WordMap = dictionary.getVisualWords(Image.open('sun_aaegrbxogokacwmz.jpg'), filterBank.filterbank().create_filterbanks(), Dictionary)
    #pdb.set_trace()
    #getImageFeatures(WordMap, len(Dictionary[0]))
    #getImageFeaturesSPM(3,WordMap, len(Dictionary[0]))
    trainSystem()
    evaluateRecognitionSystem()