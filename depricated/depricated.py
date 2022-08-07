import os # not sure if this is needed but we can delete it later
import matplotlib.pyplot as plt # we probably will not need this for the final project
import numpy as np
from skimage import io # not sure if I am allowed to use this but oh well

n_colors = 4 # number of output values. (IMPORTANT: not sure if this is what is needed for the assignment)

# load image. --> [FIXME : make this a function]
image_raw = io.imread("./imgs/DSC_1696a.tif") # remember to switch out images
image = np.array(image_raw, dtype=np.float64) / 255
h, w, d = image.shape
image_array = np.reshape(image, (h * w, d))

print(image_array)

# plot original image (this is not part of proj).
plt.figure(1)
plt.clf()
plt.axis("off")
plt.title("Original image")
plt.imshow(image)

# histogram of orginal image (this is not part of proj).
plt.figure(2)
plt.clf()
plt.title("Histogram original image")
hist = np.histogram(image_raw, bins=np.arange(0, 256))
plt.plot(hist[1][:-1], hist[0], lw=2)
plt.show()


def dct(input, mask):
    print(len(input))

dct(image_array, 2)


# create quantized image. [TODO] 
# a.	The linear quantization matrix Q = p*8./hilb(8); 
# b.	The DCT C matrix is defined on the top of page 528
# c.	The compression and decompression code on the top of page 533


# STUDY SECTION 11.2 OF THE TEXT
# https://www.youtube.com/watch?v=O0suOccKLbs
# https://www.youtube.com/watch?v=KWc9SOOLfLw
