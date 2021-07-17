# -*- coding: utf-8 -*-
"""
Created on Sat May  8 03:28:25 2021

@author: abc
"""

import numpy as np
import cv2
img = cv2.imread('BSE_Image.jpg') #it is RGB image

#reshape the image
img2 = img.reshape((-1,3))

#In opencv image type must be in float
#So that we change it datatype (int to float)
img2 = np.float32(img2)

#define some criteria
#It is the iteration termination criteria. When this criteria is satisfied, algorithm iteration stops.
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
#cv.TERM_CRITERIA_MAX_ITER - stop the algorithm after the specified number of iterations, max_iter.

#Clusters
k=4

#put attempts
attempts = 10 #it's specify number of times the algorithm is executed initial labellings

#It is going to return labels
ret, label, center=cv2.kmeans(img2,k,None,criteria,10,cv2.KMEANS_PP_CENTERS)

#flags : This flag is used to specify how initial centers are taken.
# Normally two flags are used for this : cv.KMEANS_PP_CENTERS and cv.KMEANS_RANDOM_CENTERS.

# take the centers and convert into uint
#centers : This is array of centers of clusters.
center = np.uint8(center)

#labels : This is the label array where each element marked '0', '1'.....
res = center[label.flatten()]
res2 = res.reshape((img.shape))

#Now just visualize our image
cv2.imwrite('segmented.jpg',res2)











