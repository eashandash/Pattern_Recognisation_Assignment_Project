import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
from math import sqrt
from matplotlib import pyplot as plt
from itertools import chain
from skimage import io
from scipy.fftpack import fft, ifft

# import image
image =  io.imread('images/black_leaf.png')
#display image
plt.imshow(image, cmap='Greys')
plt.show()

img=image

#invert image
(row, col) = img.shape

for i in range(row):
    for j in range(col):
        if img[i,j] == 255:
            img[i,j] = 0
        else:
            img[i,j] = 255

# ret,img = cv2.threshold(image,70,255,0)
#display image
plt.imshow(img, cmap='Greys')
plt.show()

###################### GET CHAIN CODE
for i, row in enumerate(img):
    for j, value in enumerate(row):
        if value == 255:
            start_point = (i, j)
            print(start_point, value)
            break
    else:
        continue
    break

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

border = []  # store a list of border points on the image
chain = []   # store a list of chain code for the image
curr_point = start_point
for direction in directions:
    idx = dir2idx[direction]
    new_point = (start_point[0]+change_i[idx], start_point[1]+change_j[idx])
    if img[new_point] != 0: # if is ROI
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
        if image[new_point] != 0: # if is ROI
            border.append(new_point)
            chain.append(direction)
            curr_point = new_point
            break
    #if count == 1000: break
    count += 1


#print(count)  # print number of border points of the image
#print(border) # print border points of the image
#print(chain) # print chain code of the image

# display image with the chain code/border points in blue
plt.imshow(img, cmap='Greys')
plt.plot([i[1] for i in border], [i[0] for i in border])
plt.show()


# fft
complex_border = [complex(*point) for point in border]
#print('#########' + str(len(complex_border)))
Y = fft(complex_border)
#Y = Y[:-50]
#print('#########' + str(len(Y)))
print(Y)

# IFFT
X = ifft(Y)
print(len(X))

inverse_img = np.ones(img.shape, img.dtype)*255
for x in X:
    inverse_img[int(x.real), int(x.imag)] = 0

plt.imshow(inverse_img, cmap='Greys')
plt.show()

#fourier_points = [[pt.real,pt.imag] for pt in Y]
#print(fourier_points)
