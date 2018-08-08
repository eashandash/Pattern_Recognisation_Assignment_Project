#differential chain code

#the image is converted to black and white (gray image) for simple storage and manipulation of the image 
#and the boundary is extracted to to get the chain code from the direction matrix.

#---------------------------------------------------------------------------------------------------


# coding: utf-8

# In[182]:

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
img = cv.imread('leaf.png',0)
ret,thresh1 = cv.threshold(img,232,255,cv.THRESH_BINARY)


# In[183]:

get_ipython().magic('matplotlib inline')


# In[199]:

plt.imshow(thresh1,'gray')


# In[202]:

thresh1.shape


# In[204]:

import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('leaf.png',0)
edges = cv2.Canny(thresh1,100,200)

plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

plt.show()


# In[187]:

## Discover the first point 
for i, row in enumerate(edges):
    for j, value in enumerate(row):
        if value == 255:
            start_point = (i, j)
            print(start_point, value)
            break
    else:
        continue
    break


# In[188]:

directions = [ 0,  1,  2,
               7,      3,
               6,  5,  4]
dir2idx = dict(zip(directions, range(len(directions))))

change_j =   [-1,  0,  1, # x or columns
              -1,      1,
              -1,  0,  1]

change_i =   [-1, -1, -1, # y or rows
               0,      0,
               1,  1,  1]

border = []
chain = []
curr_point = start_point
for direction in directions:
    idx = dir2idx[direction]
    new_point = (start_point[0]+change_i[idx], start_point[1]+change_j[idx])
    if edges[new_point] != 0: # if is ROI
        border.append(new_point)
        chain.append(direction)
        curr_point = new_point
        break

count = 0
while curr_point != start_point:
    #figure direction to start search
    b_direction = (direction + 5) % 8 
    dirs_1 = range(b_direction, 8)
    dirs_2 = range(0, b_direction)
    dirs = []
    dirs.extend(dirs_1)
    dirs.extend(dirs_2)
    for direction in dirs:
        idx = dir2idx[direction]
        new_point = (curr_point[0]+change_i[idx], curr_point[1]+change_j[idx])
        if edges[new_point] != 0: # if is ROI
            border.append(new_point)
            chain.append(direction)
            curr_point = new_point
            break
    if count == 1000: break
    count += 1


# In[189]:

print(count)
print(chain)




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

#the image is converted to black and white (gray image) for simple storage and manipulation of the image 
#and the boundary is extracted to to get the chain code from the direction matrix.





