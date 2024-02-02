import cv2 
import numpy as np 
from numpy.fft import fft2, ifft2, fftshift 
 

image = cv2.imread('C:\\Users\\bdbir\\Desktop\\signal\\downsample\\sprite.png', 0) 

rows, cols = image.shape 

img = cv2.resize(image, (int(cols/2), int(rows/2))) 
 
fft = np.fft.fft2(img) 

fshift = np.fft.fftshift(fft) 
 
rows, cols = img.shape 
crow, ccol = int(rows/2), int(cols/2) 
fshift[crow-30:crow+30, ccol-30:ccol+30] = 0 
 
f_ishift = np.fft.ifftshift(fshift) 
img_back = np.fft.ifft2(f_ishift) 
img_back = np.abs(img_back) 
 
combined = np.zeros((rows*2, cols*2), dtype=img.dtype) 
 
combined[rows//2:rows+rows//2, cols//2:cols+cols//2] = cv2.resize(img, (cols, rows), interpolation=cv2.INTER_AREA) 
 
 
cv2.imwrite('C:\\Users\\bdbir\\Desktop\\signal\\downsample\\downsample_image.jpg', combined) 
