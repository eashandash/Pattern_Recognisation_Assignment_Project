#Fourier Descripter of the input image
#For the input image, FFT is applied to get the fourier descripter of the image and the same is displayed 

#------------------------------------------------------------------------------------------------------------------
# coding: utf-8

# In[2]:

import numpy as np
import cv2
from matplotlib import pyplot as plt
img = cv2.imread('final_leaf.png',0)
img


# In[3]:

img.shape


# In[4]:

dft = cv2.dft(np.float32(img),flags = cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))

plt.subplot(121),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()


# In[22]:

import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('final_leaf.png',0)

dft = cv2.dft(np.float32(img),flags = cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))



# In[23]:

plt.subplot(121),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()


# In[ ]:


#test image- FEAUTURE VECTOR
#Image feature vector of class 2 image
img.shape
#raw pixel feature vector of class 2 image
raw3=img.flatten()
raw3.shape
raw3
#color mean descriptor of class 2 image
mean3 = cv2.mean(img)
mean3
#color mean and standard deviation of class 2 image
(mean3, std3) = cv2.meanStdDev(img)
mean2,std2
#combining mean and standard deviation
stat3 = np.concatenate([mean3, std3]).flatten()
stat3



# In[ ]:

#SUMMARY/CONCLUSION 

#For the input image, FFT is applied to get the fourier descripter of the image and the same is displayed  



#-------------------------------------------------------------------------------------------------



