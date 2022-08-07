import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy

from numpy import pi
from numpy import sin
from numpy import zeros
from numpy import r_
from scipy import signal
from scipy import misc # pip install Pillow
from scipy.linalg import hilbert
import matplotlib.pylab as pylab

from skimage import io
from PIL import Image

image = io.imread('./imgs/DSC_1696a.tif')
image_shape = image[80:88, 80:88, 2]
# image_shape = image.shape # rows, columns, colors

# print(image_shape)/\

# img = np.array(image_size, dtype=np.uint8)
# io.imsave('testing-image.jpg', img)
p = 1
q = (p*8)/(hilbert(8))

print(q)

img = np.array(q*image_shape, dtype=np.uint8)
io.imsave('testing-image.jpg', img)