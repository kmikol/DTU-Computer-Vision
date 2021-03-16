# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 14:36:24 2021

@author: Youyang Shen
"""

import numpy as np
import scipy.ndimage
import skimage.io
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage.color import rgb2gray

#%% Read in image data
data_path = 'week06_data/'
im_name = 'TestIm1.png'

im = skimage.io.imread(data_path + im_name).astype(int)
imgray = rgb2gray(im)

# im = im[:,:,0:3]
fig, ax = plt.subplots(1)
ax.imshow(imgray)

#%% 06A define gaussian kernel function

def gaussian1DKernels(sigma, size=4):
    if sigma == 0:
        g = np.zeros(2*size-1)
        g[size-1]=1
        dg = g
        ddg = g
    else:
        s = np.ceil(np.max([sigma*size, size]))
        x = np.arange(-s,s+1)
        x = x.reshape(x.shape + (1,))
        g = np.exp(-x**2/(2*sigma*sigma))
        g /= np.sum(g)
        dg = -x/(sigma*sigma)*g
        # ddg = -1/(sigma*sigma)*g -x/(sigma*sigma)*dg
    return g, dg

#%% 06B gaussian smoothing

def gaussian_smoothing(im,sigma):
    g,gx = gaussian1DKernels(sigma,size = 1)
    g = g.reshape([1,-1])
    gx = gx.reshape([1,-1])
    I = scipy.ndimage.convolve(im,g)
    I = scipy.ndimage.convolve(I,g)
    Ix = scipy.ndimage.convolve(I,gx)
    Iy = scipy.ndimage.convolve(I,gx.T)
    return I,Ix,Iy

g1,dg1 = gaussian1DKernels(0)

# a = np.arange(50, step=2).reshape((5,5))
# b = gaussian_filter(a, sigma=0)

I,Ix,Iy = gaussian_smoothing(imgray,2)
fig, ax = plt.subplots(1,3)
ax[0].imshow(I)
ax[1].imshow(Ix)
ax[2].imshow(Iy)

#%% 06C

def smoothedHessian(im,sigma,epsilon):
    g_e,g_e1 = gaussian1DKernels(sigma,size=epsilon) #g_epsilon
    g,gx = gaussian1DKernels(sigma)
    gy = gx.T
    I,Ix,Iy = gaussian_smoothing(im,sigma)
    Ix2 = pow(Ix,2)
    Iy2 = pow(Iy,2)
    Ixy = Ix*Iy
    n,m = im.shape
    CIxx = np.zeros([n,m])
    CIyy = np.zeros([n,m])
    CIxy = np.zeros([n,m])
    CIxx = scipy.ndimage.convolve(Ix2,g_e)
    CIyy = scipy.ndimage.convolve(Iy2,g_e)
    CIxy = scipy.ndimage.convolve(Ixy,g_e)

    return CIxx,CIyy,CIxy

C1,C2,C3 = smoothedHessian(imgray,2,4)

fig, ax = plt.subplots(1,3)
ax[0].imshow(C1)
ax[1].imshow(C2)
ax[2].imshow(C3)

#%% 06D

def  harrisMeasure(im, sigma, epsilon, k):
    C1,C2,C3 = smoothedHessian(im,sigma,epsilon)
    n,m = im.shape
    R = np.zeros([n,m])
    for i in range(n):
        for j in range(m):
            R[i,j] = C1[i,j]*C2[i,j] - C3[i,j]*C3[i,j]-k*(pow(C1[i,j]+C2[i,j],2))
    return R

R = harrisMeasure(imgray, 1, 2, 0.02)
print(np.max(R))
plt.imshow(R)
R_threshhold = np.zeros([300,300])
R_threshhold = np.where(R > 0, 1, R)
plt.imshow(R_threshhold)
# R_threshhold = np.where(R_threshhold==1, R, R_threshhold)
#%% 06E

def cornerDetector(im, sigma, epsilon, k, tau):
    R = harrisMeasure(im, sigma, epsilon, k)
    n,m = im.shape
    R_threshhold = np.zeros([n,m])
    for i in range(n):
        for j in range(m):
            if R[i,j]>tau:
                R_threshhold[i,j] = R[i,j]
    R_local_max = []
    for i in range(1,n-1):
        for j in range(1,m-1):
            if R_threshhold[i,j]>R_threshhold[i+1,j] and R_threshhold[i,j]>R_threshhold[i,j+1] and R_threshhold[i,j]>=R_threshhold[i-1,j] and R_threshhold[i,j]>=R_threshhold[i,j-1]:
                R_local_max.append([j,i])
    R_local_max = np.asarray(R_local_max)
    return R_local_max

maxpoint =  cornerDetector(imgray, 1, 2, 0.02, 0)
plt.imshow(R_threshhold)
maxpoint= maxpoint.T
# print(np.where(maxpoint==1))
plt.plot(maxpoint[0],maxpoint[1],'ro')
# print(maxpoint)

#%% 06F
import cv2
import numpy as np
import scipy.ndimage
import skimage.io
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage.color import rgb2gray

img1 = cv2.imread('week06_data/TestIm1.png',1)
imggray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
edges1 = cv2.Canny(imggray1,100,200)

plt.subplot(121),plt.imshow(img1,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges1,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

plt.show()

img2 = cv2.imread('week06_data/TestIm2.png',1)
imggray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
edges2 = cv2.Canny(imggray2,100,200)

plt.subplot(121),plt.imshow(img2,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges2,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

plt.show()

#%% 06G
edges2 = cv2.Canny(imggray2,100,200)
edges3 = cv2.Canny(imggray2,1000,1)
edges4 = cv2.Canny(imggray2,100,50)
edges5 = cv2.Canny(imggray2,10,1000)
edges6 = cv2.Canny(imggray2,1000,1000)


plt.subplot(231),plt.imshow(img2,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(232),plt.imshow(edges2,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.subplot(233),plt.imshow(edges3,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.subplot(234),plt.imshow(edges4,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.subplot(235),plt.imshow(edges5,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.subplot(236),plt.imshow(edges6,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])



plt.show()





















