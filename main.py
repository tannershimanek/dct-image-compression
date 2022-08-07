import numpy as np
from numpy import r_
from skimage import io
from scipy.linalg import hilbert
from scipy.fft import dct
from scipy.fft import idct
import matplotlib.pyplot as plt

IMAGE_PATH = "./imgs/DSC_1696a.tif"
IMAGE_NAME = IMAGE_PATH.split('/')[2].split('.')[0]
QUAL = "good" # good | usable

print(f'\nCompressing [ {IMAGE_PATH} ]')

p = 7 # 4 = "great", 7 = "good", 19 = "usable"
N = 8
Q = (p*8)/(hilbert(8))  # linear quantization matrix

dct_matrix = dct(np.eye(N), axis=1, norm='ortho')

np.set_printoptions(precision=3)
print('\n', 'Base dct matrix\n', dct_matrix)
print('\n', 'Base linear quantization matrix\n', Q)

image_raw = io.imread(IMAGE_PATH).astype(float)
image = np.array(image_raw, dtype=np.uint8)  # uint8 is an 8 bit integer

# 8 x 8 blocks for red, green, and blue before DCT
print('\nSingle 8 x 8 block of red, green, and blue before DCT at position 1112\n')
print('RED\n', image[1112:1120, 1112:1120, 0], '\n') # R
print('GREEN\n', image[1112:1120, 1112:1120, 1], '\n') # G
print('BLUE\n', image[1112:1120, 1112:1120, 2], '\n') # B

image_size = image.shape

h, w, channels = image_size
height = round(h/N-1)
width = round(w/N-1)
new_image_size = (height, width, channels)
dct_zeros = np.zeros(image_size)


def dct2d(block):
    """Get the DCT of a 2 dimensional array"""
    return dct(dct(block, axis=0, norm='ortho'), axis=1, norm='ortho')


def idct2d(block):
    """Get the IDCT of a 2 dimensional array"""
    return idct(idct(block, axis=0, norm='ortho'), axis=1, norm='ortho')


print('Compressing image..')

# 8x8 DCT on image (in-place)
for i in r_[:image_size[0]:N]:
    for j in r_[:image_size[1]:N]:
        dct_zeros[i:(i+N), j:(j+N)] = dct2d(image[i:(i+N), j:(j+N)])

# p Loss Threshold
p_threshold = p/100
dct_threshold = dct_zeros * (abs(dct_zeros) > (p_threshold*np.max(dct_zeros)))

nonzeros_percent = np.sum(dct_threshold != 0.0) / \
    (image_size[0]*image_size[1]*1.0)

print("Keeping only %f%% of the DCT coefficients.." % (nonzeros_percent*100.0))

img_dct = np.zeros(image_size)

for i in r_[:image_size[0]:N]:
    for j in r_[:image_size[1]:N]:
        img_dct[i:(i+N), j:(j+N)] = idct2d(dct_threshold[i:(i+N), j:(j+N)])

# create a new image
img = np.array(img_dct, dtype=np.float64)
img = img.astype(np.uint8)

# 8 x 8 blocks for red, green, and blue after DCT
print('\nSingle 8 x 8 block of red, green, and blue after DCT at position 1112\n')
print('RED\n', img[1112:1120, 1112:1120, 0], '\n') # R
print('GREEN\n', img[1112:1120, 1112:1120, 1], '\n') # G
print('BLUE\n', img[1112:1120, 1112:1120, 2], '\n') # B

output_name = f'./output/{IMAGE_NAME}-{QUAL}.jpg'
print(f'\nWriting file [ {output_name} ]')

io.imsave(output_name, img)

plt.figure()
plt.imshow(np.hstack((image, img)), cmap='gray')
plt.title("Comparison between original and DCT compressed images")

plt.show()

print('Done.\n')
