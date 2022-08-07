from scipy.linalg import hilbert

p=12
Q = (p*8)/(hilbert(8))  # linear quantization matrix
