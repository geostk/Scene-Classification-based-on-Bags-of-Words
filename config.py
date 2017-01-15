#import os
#import os.path as osp
import numpy as np
# `pip install easydict` if you don't have it
from easydict import EasyDict as edict

# Consumers can get config by:
#   from config import cfg
cfg = edict()

cfg.K_FOR_KMEANS = 200

cfg.IMAGES_PATH = './images'

cfg.WORDMAP_DIR = './wordmap'
cfg.NUM_ITER_FOR_KMEANS = 10

# cfg.ALPHA :The number of pixels choosen from an image to compute the dictionary
cfg.ALPHA = 200 

cfg.K_FOR_KNN = 10

cfg.SCALES_FOR_CREATINMG_FILTERBANK = range(1,4)

cfg.GAUSSIAN_SIGMAS = np.array([1, 2, 4])
cfg.LOG_SIGMAS = np.array([1, 2, 4, 8])
cfg.D_GAUSSIAN_SIGMAS = np.array([2, 4])

cfg.NUMBER_OF_LAYER_FOR_SPM = 4
USE_SMALL_DATA_SET = False
if USE_SMALL_DATA_SET:
    cfg.TRAIN_IMAGE_PATHS = 'smallTrainImagePaths'
    cfg.TEST_IMAGE_PATHS = 'smallTestImagePaths'   
    cfg.TRAIN_IMAGE_LABELS = 'smallTrainImageLabels' 
    cfg.TEST_IMAGE_LABELS = 'smallTestImageLabels' 
    
else:
    cfg.TRAIN_IMAGE_PATHS = 'trainImagePaths' 
    cfg.TEST_IMAGE_PATHS = 'testImagePaths'   
    cfg.TRAIN_IMAGE_LABELS = 'trainImageLabels' 
    cfg.TEST_IMAGE_LABELS = 'testImageLabels' 
        