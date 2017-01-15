import numpy as np
import pdb
import cPickle
from PIL import Image
import matplotlib.pyplot as plt
import scipy.io as sio
from dictionary import computeDictionary,getVisualWords
import os
from multiprocessing import Pool
from scipy import stats
from config import cfg


def getImageFeatures(WordMap,DictionarySize):
    """
    Description:
    extract the histogram of visual words within the given image
    input:
        WordMap: A H x W image containing the IDs of visual words within the 
        given image
        DictionarySize: The number of visual words in the dictionary
    return:
        The histogram of visual words within the given image
    """
    hist = np.zeros((DictionarySize,))
    numOfPixel = WordMap.size
    WordMap=np.reshape(WordMap,(numOfPixel,))
    for i in xrange(numOfPixel):
        hist[WordMap[i]]=hist[WordMap[i]]+1
    hist = hist/numOfPixel
    return hist

def getImageFeaturesSPM(LayerNum, WordMap, DictionarySize):
    """
    Description:
    Form a multi-resolution representation of a given wordMap based on the 
    Spatial Pyramid Matching
    input:
        LayerNum: Number of layers in the Spatial Pyramid
        wordMap: A H x W image containing the IDs of visual words within the 
        given image
        DictionarySize: The number of visual words in the dictionary

    return: 
        Hist: A multi-resolution representation of a given wordMap based on the 
    Spatial Pyramid Matching

    """
    L=LayerNum-1

    (H,W) = WordMap.shape
    CellHeight=H/(2**L)
    CellWidth=W/(2**L)
    LayerHist=np.zeros(((4**L)*DictionarySize,))
    SavedHist=np.zeros((4**L,DictionarySize)) # Save finer layer histograms for computing the next layer
    NewHist=np.zeros((4**L,DictionarySize)) # Save layer histograms for updating saved_hist
    for i in xrange(2**L):   
        for j in xrange(2**L):
            StartColum=i*CellWidth
            StartRow=j*CellHeight
            CurrentHist = getImageFeatures(WordMap[StartRow:StartRow+CellHeight,StartColum:StartColum+CellWidth],DictionarySize)
            LayerHist[i*(2**L)*DictionarySize+j*DictionarySize:i*(2**L)*DictionarySize+j*DictionarySize+DictionarySize]=CurrentHist
            #print 'sum(CurrentHist)=',np.sum(CurrentHist)
            #print 'sum(LayerHist)=',np.sum(LayerHist)
            SavedHist[i*(2**L)+j,:]=CurrentHist
            
# normalize
    LayerHist = LayerHist / (4**L)
    #print (4**L),'sum LayerHist = LayerHist / (4**L)= ', np.sum(LayerHist)
    if(L>1):
        weight=2**(-1)
    else:
        weight=2**(-L)
    
    Hist=LayerHist*weight



    for l in reversed(range(L)):# L-1:-1:0
        
        LayerHist=np.zeros(((4**l)*DictionarySize))
        
        for i in xrange(0,2**l):#1:2^l
            for j in xrange(0,2**l):#j=1:2^l
                NewHist[i*(2**l)+j,:] = \
                    SavedHist[0+(i)*(2**(l+2))+(j)*2,:]+ \
                    SavedHist[1+(i)*(2**(l+2))+(j)*2,:]+ \
                    SavedHist[0+(i)*(2**(l+2))+(j)*2+2**(l+1),:]+ \
                    SavedHist[1+(i)*(2**(l+2))+(j)*2+2**(l+1),:]
                
                LayerHist[i*(2**l)*DictionarySize+j*DictionarySize:i*(2**l)*DictionarySize+\
                j*DictionarySize+DictionarySize]= NewHist[(i)*(2**l)+j,:]/4
            
       
        SavedHist=NewHist/4;
        if(l>1):
            weight=2**(l-L-1);
        else:
            weight=2**(-L);
        
        #normalize
        LayerHist =LayerHist / (4**l);
        #print 'this sum =',np.sum(LayerHist)
        Hist=np.hstack((LayerHist*weight,Hist));
    #pdb.set_trace()  
    return Hist

def distanceToSet(WordHist, Histograms):
    """
    Description:
        returns the histogram intersection similarity between WordHist and each 
        training sample as a 1XT vector(T is the number of training samples)
        intput 
            WordHist:
            A  N x 1 vector represents the features of an image
            Histograms: A N x T vector contains features of all training images
        return:
            HistInter: the histogram intersection similarity between WordHist 
            and each training sample
    """
    WordHist = WordHist.reshape((-1,1))
    WordHist =np.tile(WordHist,(1,Histograms.shape[1]))

    HistInter = np.sum(np.minimum(WordHist,Histograms),axis=0)
    return HistInter

def trainSystem():
    TrainTestMatFile = sio.loadmat('traintest.mat')
    ImageDir = cfg.IMAGES_PATH
    TargetDir = cfg.WORDMAP_DIR
    TrainImagePaths = TrainTestMatFile[cfg.TRAIN_IMAGE_PATHS]
    Classnames = TrainTestMatFile['classnames']
    #pdb.set_trace()
    print "Computing dictionary ... ",
    computeDictionary(TrainImagePaths,ImageDir)
    print "Computing dictionary done... ",
    Dictionary = cPickle.load(open('dictionary.pkl', 'rb'))
    FilterBank = cPickle.load(open('filterbank.pkl', 'rb'))
    print 'Computing visual words ... ',
    batchToVisualWords(TrainImagePaths,Classnames,FilterBank,Dictionary,ImageDir,TargetDir)
    print 'Done',
    TrainHistograms=createHistograms(len(Dictionary[0]),TrainImagePaths,TargetDir)
    cPickle.dump(TrainHistograms,open('TrainHistograms.pkl', 'wb'))
    #pdb.set_trace()




def MPWorkerToGetVisualWords(x):
    """
    Worker for multiprocess to get visual words from an image and save it to the
    traget directory
    """
    ImagePath,FilterBank,Dictionary,ImageDir,TargetDir = x

    print 'Opening image:',ImagePath[0][0]
    I = Image.open(os.path.join(ImageDir,ImagePath[0][0]))
    print 'Converting to visual words {0}\n'.format(ImagePath[0][0])
    WordRepresentation = getVisualWords(I, FilterBank, Dictionary)
    OutPutPath = os.path.join(TargetDir,ImagePath[0][0]+'.pkl')
    cPickle.dump(WordRepresentation,open(OutPutPath, 'wb'))

def batchToVisualWords(TrainImagePaths,Classnames,FilterBank,Dictionary,ImageDir,
    TargetDir):
    """
    Get the visual words from all training image and save them

    """
    if not os.path.exists(TargetDir):
        os.mkdir(TargetDir)
    for c in xrange(len(Classnames)):
        temppath=os.path.join(TargetDir,Classnames[c,0][0])
        if not os.path.exists(temppath):
            os.mkdir(os.path.join(TargetDir,Classnames[c,0][0]))

    x = [(a,FilterBank,Dictionary,ImageDir,TargetDir) for a in TrainImagePaths]
    #pdb.set_trace()
    p = Pool(None)
    p.map(MPWorkerToGetVisualWords, x)
    p.close()
    p.join()

def createHistograms(DictionarySize,TrainImagePaths,TargetDir):
    """
    concatenate all the histograms from each training image into a single matrix
    """
    LayerNum=cfg.NUMBER_OF_LAYER_FOR_SPM
    ImageDir=cfg.IMAGES_PATH
    OutPutHistograms = np.zeros((DictionarySize*((4**(LayerNum))/3),len(TrainImagePaths)))
    for i in xrange(len(TrainImagePaths)):
        print 'createHistograms',i
        WordMap = cPickle.load(open(os.path.join(TargetDir,TrainImagePaths[i][0][0]+'.pkl')))
        OutPutHistograms[:,i] = getImageFeaturesSPM(LayerNum,WordMap,DictionarySize)
    #pdb.set_trace() 
    return OutPutHistograms


def evaluateRecognitionSystem():
    print "Evaluating the system..."
    ImageDir=cfg.IMAGES_PATH
    TrainTestMatFile = sio.loadmat('traintest.mat')
    TestImagePaths = TrainTestMatFile[cfg.TEST_IMAGE_PATHS]
    TrainImageLabels = TrainTestMatFile[cfg.TRAIN_IMAGE_LABELS]
    TestImageLabels  = TrainTestMatFile[cfg.TEST_IMAGE_LABELS]
    Dictionary = cPickle.load(open('dictionary.pkl', 'rb'))
    FilterBank = cPickle.load(open('filterbank.pkl', 'rb'))
    TrainHistograms = cPickle.load(open('TrainHistograms.pkl', 'rb'))
    DictionarySize=len(Dictionary[0])
    k= cfg.K_FOR_KNN
    ConfusionMatrix=np.zeros((9,9))
    LayerNum= cfg.NUMBER_OF_LAYER_FOR_SPM 
    NumOfTestImages=len(TestImagePaths)
    for i in xrange(NumOfTestImages):
        print i+1,'/',NumOfTestImages
        I = Image.open(os.path.join(ImageDir,TestImagePaths[i][0][0]))
        WordMap=getVisualWords(I,FilterBank,Dictionary)
        WordHist=getImageFeaturesSPM(LayerNum,WordMap,DictionarySize)
        predictedLabel = knnClassify(WordHist,TrainHistograms,TrainImageLabels,k)
        
        ConfusionMatrix[TestImageLabels[i][0]-1,predictedLabel-1]=ConfusionMatrix[TestImageLabels[i][0]-1,predictedLabel-1]+1;
    Accuracy = np.trace(ConfusionMatrix)/np.sum(ConfusionMatrix)
    print 'ConfusionMatrix=\n',ConfusionMatrix
    print 'Accuracy=',Accuracy

def knnClassify(WordHist,TrainHistograms,TrainImageLabels,k):
    Distances = distanceToSet(WordHist, TrainHistograms)
    I = np.argsort(Distances)[::-1]
    PredictedLabel= stats.mode(TrainImageLabels[I[0:k]])
    return PredictedLabel[0][0][0]

def visualizeWords():
    print 'Visualizing the words...'
    ImageDir=cfg.IMAGES_PATH
    TargetDir = cfg.WORDMAP_DIR
    Dictionary = cPickle.load(open('dictionary.pkl', 'rb'))
    TrainTestMatFile = sio.loadmat('traintest.mat')
    TrainImageLabels = TrainTestMatFile[cfg.TRAIN_IMAGE_LABELS]
    TrainImagePaths = TrainTestMatFile[cfg.TRAIN_IMAGE_PATHS]
    DictionarySize = len(Dictionary[0])
    RecomputePatch = True
    if RecomputePatch:
        
        PatchSize=9;
        HalfOfPatchSize=4;
        AverPatchOfWords= []
        
        for i in xrange(DictionarySize):
            AverPatchOfWords.append(np.zeros((PatchSize,PatchSize,3)))

        CountOfWords = np.zeros((DictionarySize,))
        for t in TrainImagePaths:
            I = Image.open(os.path.join(ImageDir,t[0][0]))
            WordMap =  cPickle.load(open(os.path.join(TargetDir,t[0][0]+'.pkl'), 'rb'))
            #WordMap =  cPickle.load(open(os.path.join(TargetDir,'bamboo_forest/sun_abqaamobpeckykne.jpg'+'.pkl'), 'rb'))
            W,H = I.size
            I = np.array(I)
            #print I.shape,'\n'
            if (H<PatchSize or W <PatchSize):
                continue
            for j in xrange(H):
                for k in xrange(W):
                    if (j<HalfOfPatchSize or k< HalfOfPatchSize or j>(H-HalfOfPatchSize-1) or k>(W-HalfOfPatchSize-1)):
                        continue
                    CountOfWords[WordMap[j,k]]+=1
                    AverPatchOfWords[WordMap[j,k]] = AverPatchOfWords[WordMap[j,k]] + I [j-HalfOfPatchSize:j+HalfOfPatchSize+1,k-HalfOfPatchSize:k+HalfOfPatchSize+1,:]
         
        for i in xrange(DictionarySize):
            if (CountOfWords[i]>0):
                AverPatchOfWords[i] = np.rint(AverPatchOfWords[i]/CountOfWords[i])
        cPickle.dump(AverPatchOfWords,open('AverPatchOfWords.pkl', 'wb'))
    #pdb.set_trace()
    AverPatchOfWords = cPickle.load(open('AverPatchOfWords.pkl', 'rb'))
    NumOfColumTodisplay = 20
    k =int(np.ceil(DictionarySize/float(NumOfColumTodisplay))) # number of words in each row to display
    f, axarr=plt.subplots(k,NumOfColumTodisplay)

    for i in xrange(DictionarySize):
        # draw the words
        a = axarr[int(i/NumOfColumTodisplay),int(i%NumOfColumTodisplay)]
        a.imshow(AverPatchOfWords[i])
        a.set_xticks([])
        a.set_yticks([])
   
    for i in xrange(DictionarySize,k*NumOfColumTodisplay):
        # fill the empyt sub figure
        a = axarr[int(i/NumOfColumTodisplay),int(i%NumOfColumTodisplay)]
        a.imshow(np.zeros(AverPatchOfWords[0].shape))
        a.set_xticks([])
        a.set_yticks([])
    f.subplots_adjust(hspace=0,wspace=0)  
    plt.show()    

if __name__ == '__main__':
    trainSystem()
    evaluateRecognitionSystem()
    visualizeWords()