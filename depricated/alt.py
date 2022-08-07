import numpy as np
from numpy import *
import matplotlib.pyplot as plt
from matplotlib.pyplot import *
import matplotlib.image as mpimg
import math



# The 8 x 8 DCT matrix
N = 8
dct = np.zeros((N, N))
for x in range(N):
    dct[0,x] = math.sqrt(2.0/N) / math.sqrt(2.0)
for u in range(1,N):
    for x in range(N):
        dct[u,x] = math.sqrt(2.0/N) * math.cos((math.pi/N) * u * (x + 0.5) )
        
np.set_printoptions(precision=3)
print(dct)

img = mpimg.imread('./imgs/DSC_1696a.tif')
plt.imshow(img)
# plt.show()

# print(img.shape)

tiny = img[40:48, 40:48, 0]

print(tiny)

def doDCT(grid):
    return dot(dot(dct, grid), dct_transpose)

def show_image(img):
    plt.imshow(img)
    plt.colorbar()
    # plt.show()


show_image(tiny)


# https://cs.marlboro.college/cours/spring2019/algorithms/code/discrete_cosine_transform/dct.html
# https://inst.eecs.berkeley.edu/~ee123/sp16/Sections/JPEG_DCT_Demo.html
# 
