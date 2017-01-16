# Bag-of-Words-for-Scene-Classification

Scene classification based on Bag-of-Words and Spatial Pyramid Matching. 
Multiprocess is used to accelerate extracting features from images and Mapping
images to their visual words representation. There are 9 kinds of scene in our 
data.

config.py: Contains the configuration of some of the parameters used in this 
project.

dictionary.py: Contains functions related to creating filterbank, computing
dictionary and get the visual word map of an image.

filterbank.py: Contains the definition of some filters and the corresponding 
function to get filter responses from a given image.

RecognitionSystem.py: Contains the function for training and evaluating the system
and function for visualizing the words in dictionary

Some results:
When set the
cfg.K_FOR_KMEANS = 200
cfg.NUM_ITER_FOR_KMEANS = 10
cfg.ALPHA = 500 
cfg.K_FOR_KNN = 10
cfg.NUMBER_OF_LAYER_FOR_SPM = 4

Can get a result like:
ConfusionMatrix=

 [ 29.   0.   0.   0.   0.   0.   0.   0.   0.]

 [  0.  26.   3.   1.   4.   0.   0.   0.   1.]

 [  4.   7.   9.   3.   2.   2.   1.   1.   2.]

 [  4.   1.   0.  23.   2.   0.  10.   0.   0.]

 [  2.   0.   0.   2.  36.   0.   4.   0.   0.]

 [  2.   8.   7.   2.   2.   2.   1.   0.   2.]

 [  1.   0.   1.   2.  13.   0.  17.   0.   0.]

 [  5.   4.   1.   2.   3.   0.   1.  15.   0.]

 [  1.   3.   1.   0.   2.   0.   0.   0.  44.]

 Accuracy= 0.626168224299
 
|#|image| word map|
|1|<img src ="https://github.com/skfory/Bag-of-Words-for-Scene-Classification/blob/master/result_image/image1.jpg"  width="400" height = "400"/>|
<img src ="https://github.com/skfory/Bag-of-Words-for-Scene-Classification/blob/master/result_image/WordMap_1.png"  width="400" height = "400" />|
|2|
<img src ="https://github.com/skfory/Bag-of-Words-for-Scene-Classification/blob/master/result_image/image2.jpg"  width="400" height = "400" />|
<img src ="https://github.com/skfory/Bag-of-Words-for-Scene-Classification/blob/master/result_image/WordMap_2.png"  width="400" height = "400" />
|3|
<img src ="https://github.com/skfory/Bag-of-Words-for-Scene-Classification/blob/master/result_image/image3.jpg"  width="400" height = "400" />|
<img src ="https://github.com/skfory/Bag-of-Words-for-Scene-Classification/blob/master/result_image/WordMap_3.png"  width="400" height = "400" />|

Visualization of the words in dictionary:
<img src ="https://github.com/skfory/Bag-of-Words-for-Scene-Classification/blob/master/result_image/words.png"  />

