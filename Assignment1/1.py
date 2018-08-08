#classifying the test image into class1 or class2 image.

#GLCM is used for the 2nd order texture analysise to classify the test image into class1 or class2 images.

#the GLCM of test image is closer to Class 2 image .
#and also the texture features such as dissimilarity, contrast, homogenity , ASM, energy of test image and class 1 and
#class2 images are compared and scatter graph between contrast and dissimilarity are ploted for class images and test image is 
#found to be closer to class 1 image.  

#--------------------------------------------------------------------------------------------------


# coding: utf-8

# In[77]:

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import greycomatrix, greycoprops
from skimage import data
get_ipython().magic('matplotlib inline')


# In[78]:

brown = cv2.imread('1_brown.png', 0)


# In[79]:

brown


# In[80]:

plt.imshow(brown, cmap = 'gray')


# In[81]:

plt.imshow(brown)


# In[82]:

brown.shape


# In[83]:

result = greycomatrix(brown, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4],
                       levels=256)


#obtaining the gray level co occurance matix of class 1 image.
#----------------------------------------------------------------



# In[122]:

contrast_brown = greycoprops(result, 'contrast')
dissimilarity_brown = greycoprops(result, 'dissimilarity')
homogeneity_brown = greycoprops(result, 'homogeneity')
ASM_brown = greycoprops(result, 'ASM')
energy_brown = greycoprops(result, 'energy')

#obtaining texture features from the GLCM of class 1
#------------------------------------------------------------------



# In[52]:

result[:, :, 0, 1]


# In[141]:

plt.figure(figsize=(14,8))

plt.subplot(141),plt.imshow(brown)
plt.title('class1')

plt.subplot(142),plt.imshow(contrast_brown)
plt.title('contrast_brown')

plt.subplot(143),plt.imshow(dissimilarity_brown)
plt.title('dissimilarity_brown')

plt.subplot(144),plt.imshow(homogeneity_brown)
plt.title('homogenity_brown')

plt.figure(figsize=(14,8))

plt.subplot(121),plt.imshow(ASM_brown)
plt.title('ASM_brown')

plt.subplot(122),plt.imshow(energy_brown)
plt.title('energy_brown')

plt.show()


# In[94]:

green = cv2.imread('2_green.png', 0)


# In[95]:

green.shape


# In[96]:

green


# In[97]:

result1 = greycomatrix(green, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4],
                       levels=256)


#obtaining the gray level co occurance matix of class 2 image.
#----------------------------------------------------------------




# In[20]:

result1[:, :, 0, 1][53,85]


# In[124]:

contrast_green = greycoprops(result1, 'contrast')
dissimilarity_green = greycoprops(result1, 'dissimilarity')
homogeneity_green = greycoprops(result1, 'homogeneity')
ASM_green = greycoprops(result1, 'ASM')
energy_green = greycoprops(result1, 'energy')

#obtaining texture features from the GLCM of class 2
#------------------------------------------------------------------





# In[129]:

plt.figure(figsize=(14,8))

plt.subplot(131),plt.imshow(contrast_green,cmap = 'gray')
plt.title('contrast_green')

plt.subplot(132),plt.imshow(dissimilarity_green,cmap = 'gray')
plt.title('dissimilarity_green')

plt.subplot(133),plt.imshow(homogeneity_green,cmap = 'gray')
plt.title('homogenity_green')

plt.figure(figsize=(14,8))

plt.subplot(121),plt.imshow(ASM_green,cmap = 'gray')
plt.title('ASM_green')

plt.subplot(122),plt.imshow(energy_green,cmap = 'gray')
plt.title('energy_green')

plt.show()


# In[142]:

plt.figure(figsize=(14,8))

plt.subplot(141),plt.imshow(green)
plt.title('class2')

plt.subplot(142),plt.imshow(contrast_green)
plt.title('contrast_green')

plt.subplot(143),plt.imshow(dissimilarity_green)
plt.title('dissimilarity_green')

plt.subplot(144),plt.imshow(homogeneity_green)
plt.title('homogenity_green')

plt.figure(figsize=(14,8))

plt.subplot(121),plt.imshow(ASM_green)
plt.title('ASM_green')

plt.subplot(122),plt.imshow(energy_green)
plt.title('energy_green')

plt.show()


# In[120]:

plt.scatter(contrast1, dissimilarity1)
plt.xlabel('contrast')
plt.ylabel('dissimilarity')
plt.show()


# In[106]:

classify_green = cv2.imread('3_classify_green.png', 0)


# In[107]:

classify_green.shape


# In[108]:

classify_green


# In[111]:

result2 = greycomatrix(classify_green, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4],
                       levels=256)



#obtaining the gray level co occurance matix of the test image.
#----------------------------------------------------------------



# In[29]:

result2[:, :, 0, 1][105,107]


# In[76]:

result2[105,107, 0, 1]


# In[74]:

result2.shape


# In[136]:

contrast_classify_green = greycoprops(result2, 'contrast')
dissimilarity_classify_green = greycoprops(result2, 'dissimilarity')
homogeneity_classify_green = greycoprops(result2, 'homogeneity')
ASM_classify_green = greycoprops(result2, 'ASM')
energy_classify_green = greycoprops(result2, 'energy')

#obtaining texture features from the GLCM of the test image
#------------------------------------------------------------------



# In[144]:

plt.figure(figsize=(14,8))

plt.subplot(141),plt.imshow(brown)
plt.title('class1')

plt.subplot(142),plt.imshow(contrast_brown)
plt.title('contrast_brown')

plt.subplot(143),plt.imshow(dissimilarity_brown)
plt.title('dissimilarity_brown')

plt.subplot(144),plt.imshow(homogeneity_brown)
plt.title('homogenity_brown')

plt.figure(figsize=(14,8))

plt.subplot(121),plt.imshow(ASM_brown)
plt.title('ASM_brown')

plt.subplot(122),plt.imshow(energy_brown)
plt.title('energy_brown')

plt.figure(figsize=(14,8))

plt.subplot(141),plt.imshow(green)
plt.title('class2')

plt.subplot(142),plt.imshow(contrast_green)
plt.title('contrast_green')

plt.subplot(143),plt.imshow(dissimilarity_green)
plt.title('dissimilarity_green')

plt.subplot(144),plt.imshow(homogeneity_green)
plt.title('homogenity_green')

plt.figure(figsize=(14,8))

plt.subplot(121),plt.imshow(ASM_green)
plt.title('ASM_green')

plt.subplot(122),plt.imshow(energy_green)
plt.title('energy_green')


plt.figure(figsize=(14,8))

plt.subplot(141),plt.imshow(classify_green)
plt.title('test image')

plt.subplot(142),plt.imshow(contrast_classify_green)
plt.title('contrast_classify_green')

plt.subplot(143),plt.imshow(dissimilarity_classify_green)
plt.title('dissimilarity_green')

plt.subplot(144),plt.imshow(homogeneity_classify_green)
plt.title('homogenity_classify_green')

plt.figure(figsize=(14,8))

plt.subplot(121),plt.imshow(ASM_classify_green)
plt.title('ASM_classify_green')

plt.subplot(122),plt.imshow(energy_classify_green)
plt.title('energy_classify_green')

plt.show()


# In[148]:

plt.figure(figsize=(14,6))

plt.subplot(131),plt.scatter(contrast_brown, dissimilarity_brown)
plt.title('contrast_brown vs dissimilarity_brown')

plt.subplot(132),plt.scatter(contrast_green, dissimilarity_green)
plt.title('contrast_green vs dissimilarity_green')

plt.subplot(133),plt.scatter(contrast_classify_green, dissimilarity_classify_green)
plt.title('contrast_classify_green vs dissimilarity_classify_green')

plt.show()


# In[116]:

plt.figure(figsize=(10,6))

plt.subplot(141),plt.imshow(result[:, :, 0, 0],cmap = 'gray')
plt.title('brown 0 angle')

plt.subplot(142),plt.imshow(result[:, :, 0, 1],cmap = 'gray')
plt.title('brown 45 angle')

plt.subplot(143),plt.imshow(result[:, :, 0, 2],cmap = 'gray')
plt.title('brown 90 angle')

plt.subplot(144),plt.imshow(result[:, :, 0, 3],cmap = 'gray')
plt.title('brown 135 angle')

plt.figure(figsize=(10,6))

plt.subplot(141),plt.imshow(result1[:, :, 0, 0],cmap = 'gray')
plt.title('green 0 angle')

plt.subplot(142),plt.imshow(result1[:, :, 0, 1],cmap = 'gray')
plt.title('green 45 angle')

plt.subplot(143),plt.imshow(result1[:, :, 0, 2],cmap = 'gray')
plt.title('green 90 angle')

plt.subplot(144),plt.imshow(result1[:, :, 0, 3],cmap = 'gray')
plt.title('green 135 angle')

plt.figure(figsize=(10,6))

plt.subplot(141),plt.imshow(result2[:, :, 0, 0],cmap = 'gray')
plt.title('classify_green 0 angle')

plt.subplot(142),plt.imshow(result2[:, :, 0, 1],cmap = 'gray')
plt.title('classify_green 45 angle')

plt.subplot(143),plt.imshow(result2[:, :, 0, 2],cmap = 'gray')
plt.title('classify_green 90 angle')

plt.subplot(144),plt.imshow(result2[:, :, 0, 3],cmap = 'gray')
plt.title('classify_green 135 angle')

plt.show()


# In[115]:

plt.figure(figsize=(10,6))

plt.subplot(141),plt.imshow(result[:, :, 0, 0])
plt.title('brown 0 angle')

plt.subplot(142),plt.imshow(result[:, :, 0, 1])
plt.title('brown 45 angle')

plt.subplot(143),plt.imshow(result[:, :, 0, 2])
plt.title('brown 90 angle')

plt.subplot(144),plt.imshow(result[:, :, 0, 3])
plt.title('brown 135 angle')

plt.figure(figsize=(10,6))

plt.subplot(141),plt.imshow(result1[:, :, 0, 0])
plt.title('green 0 angle')

plt.subplot(142),plt.imshow(result1[:, :, 0, 1])
plt.title('green 45 angle')

plt.subplot(143),plt.imshow(result1[:, :, 0, 2])
plt.title('green 90 angle')

plt.subplot(144),plt.imshow(result1[:, :, 0, 3])
plt.title('green 135 angle')

plt.figure(figsize=(10,6))

plt.subplot(141),plt.imshow(result2[:, :, 0, 0])
plt.title('classify_green 0 angle')

plt.subplot(142),plt.imshow(result2[:, :, 0, 1])
plt.title('classify_green 45 angle')

plt.subplot(143),plt.imshow(result2[:, :, 0, 2])
plt.title('classify_green 90 angle')

plt.subplot(144),plt.imshow(result2[:, :, 0, 3])
plt.title('classify_green 135 angle')

plt.show()


# In[ ]:

#Class 1 - FEAUTURE VECTOR
#Image feature vector of class 1 image
brown.shape
#raw pixel feature vector of class 1 image
raw1=brown.flatten()
raw1.shape
raw1
#color mean descriptor of class 1 image
mean1 = cv2.mean(brown)
mean1
#color mean and standard deviation of class 1 image
(mean1, std1) = cv2.meanStdDev(brown)
mean1,std1
#combining mean and standard deviation
stat1 = np.concatenate([mean1, std1]).flatten()
stat1




#Class 2 - FEAUTURE VECTOR
#Image feature vector of class 2 image
green.shape
#raw pixel feature vector of class 2 image
raw2=green.flatten()
raw2.shape
raw2
#color mean descriptor of class 2 image
mean2 = cv2.mean(green)
mean2
#color mean and standard deviation of class 2 image
(mean2, std2) = cv2.meanStdDev(green)
mean2,std2
#combining mean and standard deviation
stat2 = np.concatenate([mean2, std2]).flatten()
stat2

#test image- FEAUTURE VECTOR
#Image feature vector of class 2 image
classify_green.shape
#raw pixel feature vector of class 2 image
raw3=classify_green.flatten()
raw3.shape
raw3
#color mean descriptor of class 2 image
mean3 = cv2.mean(classify_green)
mean3
#color mean and standard deviation of class 2 image
(mean3, std3) = cv2.meanStdDev(classify_green)
mean2,std2
#combining mean and standard deviation
stat3 = np.concatenate([mean3, std3]).flatten()
stat3







































#SUMMARY/CONCLUSION

#GLCM is used for the 2nd order texture analysise to classify the test image into class1 or class2 images.

#the GLCM of test image is closer to Class 2 image .
#and also the texture features such as dissimilarity, contrast, homogenity , ASM, energy of test image and class 1 and
#class2 images are compared and scatter graph between contrast and dissimilarity are ploted for class images and test image is 
#found to be closer to class 1 image.  



#-----------------------------------------------------------------------------------------------