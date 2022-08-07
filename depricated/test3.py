import numpy as np
from numpy import r_
from skimage import io
from scipy.linalg import hilbert
from scipy.fft import dct
from scipy.fft import idct
import matplotlib.pyplot as plt

IMAGE_PATH = "./imgs/DSC_1696a.tif"

p = 22 # 10 = "good", 22 = "usable"
N = 8
Q = (p*8)/(hilbert(8)) # linear quantization matrix

dct_matrix = dct(np.eye(N), axis=1, norm='ortho')

np.set_printoptions(precision=3)
print(dct_matrix)
print('\n', Q)

# image_raw = io.imread(IMAGE_PATH)
image_raw = io.imread(IMAGE_PATH).astype(float)
image = np.array(image_raw, dtype=np.uint8) #/ 255 # uint8 is an 8 bit integer

image_size = image.shape

h, w, channels = image_size
height = round(h/8-1)
width = round(w/8-1)

print('\n', height, width, channels, '\n')

new_image_size = (height, width, channels)
dct_zeros = np.zeros(image_size)


def dct2(block):
    return dct(dct(block, axis=0, norm='ortho'), axis=1, norm='ortho')


def idct2(block):
    return idct(idct( block, axis=0, norm='ortho'), axis=1, norm='ortho')


# 8x8 DCT on image (in-place)
for i in r_[:image_size[0]:N]:
    for j in r_[:image_size[1]:N]:
        dct_zeros[i:(i+N),j:(j+N)] = dct2(image[i:(i+N),j:(j+N)])

# Loss Threshold

p_thresh = p/100
dct_thresh = dct_zeros * (abs(dct_zeros) > (p_thresh*np.max(dct_zeros)))

percent_nonzeros = np.sum( dct_thresh != 0.0 ) / (image_size[0]*image_size[1]*1.0)

print("Keeping only %f%% of the DCT coefficients" % (percent_nonzeros*100.0))

im_dct = np.zeros(image_size)

for i in r_[:image_size[0]:8]:
    for j in r_[:image_size[1]:8]:
        im_dct[i:(i+8),j:(j+8)] = idct2( dct_thresh[i:(i+8),j:(j+8)] )


# create a new image
img = np.array(im_dct, dtype=np.float64)
img = img.astype(np.uint8)
io.imsave('testing-image.jpg', img)

plt.figure()
plt.imshow( np.hstack( (image, img) ) ,cmap='gray')
plt.title("Comparison between original and DCT compressed images" )

plt.show()

# image_raw = io.imread("./imgs/DSC_1696a.tif") # todo: remember to switch out images
# image = np.array(image_raw, dtype=np.uint8) / 255 # uint8 is an 8 bit integer

# image_size = image.shape

# print(image_size)

