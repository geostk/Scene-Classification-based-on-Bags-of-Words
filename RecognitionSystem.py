import numpy as np
import pdb
import cPickle
import dictionary 
import filterBank
from PIL import Image
def getImageFeatures(wordMap,dictionarySize):
    pdb.set_trace()
    hist = np.zeros((dictionarySize,1))
    numOfPixel = wordMap.size
    wordMap=np.reshape(wordMap,(numOfPixel,))
    for i in xrange(numOfPixel):
        hist[wordMap[i]]=hist[wordMap[i]]+1
    pdb.set_trace()

    return hist/num_of_pixel

if __name__ == '__main__':
    Dictionary = cPickle.load(open('dictionary.pkl', 'rb'))
    wordMap = dictionary.getVisualWords(Image.open('sun_aaegrbxogokacwmz.jpg'), filterBank.filterbank().create_filterbanks(), Dictionary)
    pdb.set_trace()
    getImageFeatures(wordMap, len(Dictionary[0]))