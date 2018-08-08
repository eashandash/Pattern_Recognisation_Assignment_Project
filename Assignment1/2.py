#Classifying the test image as the subset of class image

#GLCM is used for the 2nd order texture analysise to classify the test image as a subset of the class image
#Various feataures such as dissimilarity, contrast, homogenity , ASM, energy of test image and class images and 
#scatter graph between contrast and dissimilarity are ploted for class images and test image .
#the plots and the features suggest that the test image is a subset of the class image.

#-------------------------------------------------------------------------------------------------



# coding: utf-8

# In[2]:

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import greycomatrix, greycoprops
from skimage import data
get_ipython().magic('matplotlib inline')
import matplotlib.image as mpimg


# In[12]:

image_grey_original = cv2.imread('image_grey_original.png', 0)
image_grey_1 = cv2.imread('image_grey_1.png', 0)
image_grey_2 = cv2.imread('image_grey_2.png', 0)


# In[13]:

image_grey_original.shape


# In[14]:

result = greycomatrix(image_grey_orginal, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4],
                       levels=256)
result1 = greycomatrix(image_grey_1, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4],
                       levels=256)
result2 = greycomatrix(image_grey_2, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4],
                       levels=256)


# In[15]:

contrast_image_grey_orginal = greycoprops(result, 'contrast')
dissimilarity_image_grey_orginal = greycoprops(result, 'dissimilarity')
homogeneity_image_grey_orginal = greycoprops(result, 'homogeneity')
ASM_image_grey_orginal = greycoprops(result, 'ASM')
energy_image_grey_orginal = greycoprops(result, 'energy')

contrast_image_grey_1 = greycoprops(result1, 'contrast')
dissimilarity_image_grey_1 = greycoprops(result1, 'dissimilarity')
homogeneity_image_grey_1 = greycoprops(result1, 'homogeneity')
ASM_image_grey_1 = greycoprops(result1, 'ASM')
energy_image_grey_1 = greycoprops(result1, 'energy')

contrast_image_grey_2 = greycoprops(result2, 'contrast')
dissimilarity_image_grey_2 = greycoprops(result2, 'dissimilarity')
homogeneity_image_grey_2 = greycoprops(result2, 'homogeneity')
ASM_image_grey_2 = greycoprops(result2, 'ASM')
energy_image_grey_2 = greycoprops(result2, 'energy')


# In[16]:

plt.figure(figsize=(14,8))

plt.subplot(141),plt.imshow(image_grey_orginal)
plt.title('original')

plt.subplot(142),plt.imshow(contrast_image_grey_orginal)
plt.title('contrast_original')

plt.subplot(143),plt.imshow(dissimilarity_image_grey_orginal)
plt.title('dissimilarity_original')

plt.subplot(144),plt.imshow(homogeneity_image_grey_orginal)
plt.title('homogenity_original')

plt.figure(figsize=(14,8))

plt.subplot(121),plt.imshow(ASM_image_grey_orginal)
plt.title('ASM_original')

plt.subplot(122),plt.imshow(energy_image_grey_orginal)
plt.title('energy_original')

plt.figure(figsize=(14,8))

plt.subplot(141),plt.imshow(image_grey_1)
plt.title('test image 1')

plt.subplot(142),plt.imshow(contrast_image_grey_1)
plt.title('contrast_image_grey_1')

plt.subplot(143),plt.imshow(dissimilarity_image_grey_1)
plt.title('dissimilarity_image_grey_1')

plt.subplot(144),plt.imshow(homogeneity_image_grey_1)
plt.title('homogenity_image_grey_1')

plt.figure(figsize=(14,8))

plt.subplot(121),plt.imshow(ASM_image_grey_1)
plt.title('ASM_image_grey_1')

plt.subplot(122),plt.imshow(energy_image_grey_1)
plt.title('energy_image_grey_1')


plt.figure(figsize=(14,8))

plt.subplot(141),plt.imshow(image_grey_2)
plt.title('test image 2')

plt.subplot(142),plt.imshow(contrast_image_grey_2)
plt.title('contrast_image_grey_2')

plt.subplot(143),plt.imshow(dissimilarity_image_grey_2)
plt.title('dissimilarity_image_grey_2')

plt.subplot(144),plt.imshow(homogeneity_image_grey_2)
plt.title('homogenity_image_grey_2')

plt.figure(figsize=(14,8))

plt.subplot(121),plt.imshow(ASM_image_grey_2)
plt.title('ASM_image_grey_2')

plt.subplot(122),plt.imshow(energy_image_grey_2)
plt.title('energy_image_grey_2')

plt.show()


# In[18]:

plt.figure(figsize=(14,6))

plt.subplot(131),plt.scatter(contrast_image_grey_orginal, dissimilarity_image_grey_orginal)
plt.title('contrast_orginal vs dissimilarity_orginal')

plt.subplot(132),plt.scatter(contrast_image_grey_1, dissimilarity_image_grey_1)
plt.title('contrast_test1 vs dissimilarity_test1')

plt.subplot(133),plt.scatter(contrast_image_grey_2, dissimilarity_image_grey_2)
plt.title('contrast_test2 vs dissimilarity_test2')

plt.show()


#Class 1 - FEAUTURE VECTOR
#Image feature vector of class 1 image
image_grey_original.shape
#raw pixel feature vector of class 1 image
raw1=image_grey_original.flatten()
raw1.shape
raw1
#color mean descriptor of class 1 image
mean1 = cv2.mean(image_grey_original)
mean1
#color mean and standard deviation of class 1 image
(mean1, std1) = cv2.meanStdDev(image_grey_original)
mean1,std1
#combining mean and standard deviation
stat1 = np.concatenate([mean1, std1]).flatten()
stat1




#Class 2 - FEAUTURE VECTOR
#Image feature vector of class 2 image
image_grey_1.shape
#raw pixel feature vector of class 2 image
raw2=image_grey_1.flatten()
raw2.shape
raw2
#color mean descriptor of class 2 image
mean2 = cv2.mean(image_grey_1)
mean2
#color mean and standard deviation of class 2 image
(mean2, std2) = cv2.meanStdDev(image_grey_1)
mean2,std2
#combining mean and standard deviation
stat2 = np.concatenate([mean2, std2]).flatten()
stat2

#test image- FEAUTURE VECTOR
#Image feature vector of class 2 image
image_grey_2.shape
#raw pixel feature vector of class 2 image
raw3=image_grey_2.flatten()
raw3.shape
raw3
#color mean descriptor of class 2 image
mean3 = cv2.mean(image_grey_2)
mean3
#color mean and standard deviation of class 2 image
(mean3, std3) = cv2.meanStdDev(image_grey_2)
mean2,std2
#combining mean and standard deviation
stat3 = np.concatenate([mean3, std3]).flatten()
stat3



#SUMMARY/CONCLUSION

#GLCM is used for the 2nd order texture analysise to classify the test image as a subset of the class image
#Various feataures such as dissimilarity, contrast, homogenity , ASM, energy of test image and class images and 
#scatter graph between contrast and dissimilarity are ploted for class images and test image .
#the plots and the features suggest that the test image is a subset of the class image.

#------------------------------------------------------------------------------------------------------------------