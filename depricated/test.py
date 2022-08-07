import numpy as np
import matplotlib.pyplot as plt
import scipy

from numpy import pi
from numpy import sin
from numpy import zeros
from numpy import r_
from scipy import signal
from scipy import misc # pip install Pillow
import matplotlib.pylab as pylab

from skimage import io # not sure if I am allowed to use this but oh well
from PIL import Image

import numpy as np
from numpy import *
import matplotlib.pyplot as plt
from matplotlib.pyplot import *
import matplotlib.image as mpimg
import math

# im = misc.imread("zelda.tif").astype(float)

# image_raw = io.imread("./imgs/DSC_1696a.tif") # remember to switch out images
image_raw = io.imread("./imgs/DSC_1696a.tif").astype(float)
image = np.array(image_raw, dtype=np.float64) #/ 255

image_size = image.shape

print(image_size)

# f = plt.figure()
# plt.imshow(image)

def dct2(a):
    return scipy.fftpack.dct( scipy.fftpack.dct( a, axis=0, norm='ortho' ), axis=1, norm='ortho' )

def idct2(a):
    return scipy.fftpack.idct( scipy.fftpack.idct( a, axis=0 , norm='ortho'), axis=1 , norm='ortho')

dct = np.zeros(image_size)

# Do 8x8 DCT on image (in-place)
for i in r_[:image_size[0]:8]:
    for j in r_[:image_size[1]:8]:
        dct[i:(i+8),j:(j+8)] = dct2( image[i:(i+8),j:(j+8)] )

# pos = 128

# Threshold
# thresh = 0.07 # usable
# thresh = 0.06 # usable
# thresh = 0.04 # usable
thresh = 0.012
dct_thresh = dct * (abs(dct) > (thresh*np.max(dct)))

percent_nonzeros = np.sum( dct_thresh != 0.0 ) / (image_size[0]*image_size[1]*1.0)

print( "Keeping only %f%% of the DCT coefficients" % (percent_nonzeros*100.0))

image_dct = np.zeros(image_size)

for i in r_[:image_size[0]:8]:
    for j in r_[:image_size[1]:8]:
        image_dct[i:(i+8),j:(j+8)] = idct2( dct_thresh[i:(i+8),j:(j+8)] )
        
        
plt.figure()
plt.imshow( np.hstack( (image, image_dct) ))
plt.title("Comparison between original and DCT compressed images" )

plt.show()


img = np.array(image_dct, dtype=np.float64)
# img = img.astype(np.uint8)
io.imsave('testing-image.jpg', img)
# print(type(image_dct))


# img = np.array(image_dct, dtype=np.uint8)
# io.imsave('testing-image.jpg', img)

# https://inst.eecs.berkeley.edu/~ee123/sp16/Sections/JPEG_DCT_Demo.html
# https://github.com/cxy1997/Digital-Image-Processing-Algorithms/blob/master/Project%207/src/transform_image_compression.m