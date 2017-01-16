# Bag-of-Words-for-Scene-Classification

Scene classification based on Bag-of-Words and Spatial Pyramid Matching. 
Multiprocess is used to accelerate extracting features from images and Mapping
images to their visual words representation.

config.py: Contains the configuration of some of the parameters used in this 
project.

dictionary.py: Contains functions related to creating filterbank, computing
dictionary and get the visual word map of an image.

filterbank.py: Contains the definition of some filters and the corresponding 
function to get filter responses from a given image.

RecognitionSystem.py: Contains the function for training and evaluating the system
and function for visualizing the words in dictionary

Some results:

<img src ="https://github.com/skfory/Bag-of-Words-for-Scene-Classification/blob/master/result_image/image1.jpg"  width="400" height = "400" />
<img src ="https://github.com/skfory/Bag-of-Words-for-Scene-Classification/blob/master/result_image/WordMap_1.png"  width="400" height = "400" />


