import numpy as np
import cv2
import matplotlib.pyplot as plt

path = "flowers.jpg"
image = cv2.imread(path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = cv2.resize(image, (4096, 4096))

#looking at the image as a tensor, compute compression via mps

t1 = np.reshape(image, [2]*24)
indices = [int(i/2) if i % 2 == 0 else int(i/2) + 12 for i in range(24)]
print(indices)
t2 = np.transpose(t1, indices)
t3 = np.reshape(t2, [4]*12)

#mps compression

us = []
inp = t3
for k in range(1, 12):
    print("k: ",  k)
    print("inp: ", inp.shape)
    inp = np.reshape(inp, [-1, 4**(12-k)])

    U, S, V = np.linalg.svd(inp, full_matrices=False)
    #take only first 100 singular values
    U = U[:, :100]
    S = S[:100]
    V = V[:100, :]
    U = U.reshape([-1, 4, len(S)])
    us.append(U)
    print("U: ", U.shape)
    print("S: ", S.shape)
    print("V: ", V.shape)
    rest = np.diag(S) @ V
    inp = rest

inp = inp.reshape([-1, 4, 1])
us.append(inp)
#inp = rest.reshape()



#from mps to t3prime
init = np.array([[1]])
for alf in range(12):
    init = np.einsum("...i,ijk->...jk", init, us[alf])
print(init.shape)


#reverse transformations
t3prime = init #t3
t2prime = np.reshape(t3prime, [2]*24)
indices1 = [i*2 for i in range(12)]
indices2 = [i*2 + 1 for i in range(12)]
indices1.extend(indices2)
print(indices1)
t1prime = np.transpose(t2prime, indices1)
t0prime = np.reshape(t1prime, [4096]*2)

plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.subplot(1, 2, 2)
plt.imshow(t0prime, cmap='gray')
plt.show()

#calculate simple SVD on the image
#keep first 100 singular values
parameters = 0
for i in range(len(us)):
    parameters += np.prod(us[i].shape)


parames_svd = int((parameters/2) / 4096)


U, S, V = np.linalg.svd(image, full_matrices=False)
U = U[:, :parames_svd]
S = S[:parames_svd]
V = V[:parames_svd, :]

#reconstruct image
rest = np.diag(S) @ V
rest = U @ rest

plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.subplot(1, 2, 2)
plt.imshow(rest, cmap='gray')
plt.show()

#compare the two images
plt.subplot(1, 2, 1)
plt.title("MPS")
plt.imshow(t0prime, cmap='gray')
plt.subplot(1, 2, 2)
plt.title("SVD")
plt.imshow(rest, cmap='gray')
plt.show()

#calculate the number of parameters in the MPS
parameters = 0
for i in range(len(us)):
    parameters += np.prod(us[i].shape)

parames_svd = (parameters/2) / 4096
print("Parameters in MPS: ", parameters)
print("Parameters in SVD: ", parames_svd)



#calculate the compression factor as a function of the bond dimension (us dimension)
parameters = 0
for i in range(len(us)):
    parameters += us[i].shape[0] * us[i].shape[1] * us[i].shape[2]
#parameters += us[-1].shape[0] * us[-1].shape[1]
factor = parameters / (4096 * 4096)

#calculate compression factor for svd
factor_svd = (parames_svd*2) / 4096

#compression factor should be the same


#compare the errors obtained with the mps and svd compression as a function of the compression factor

#calculate the error of the svd compression
error_svd = np.linalg.norm(image - rest) / np.linalg.norm(image)

#calculate the error of the mps compression
error_mps = np.linalg.norm(image - t0prime) / np.linalg.norm(image)

#compare the two errors as a function of the compression factor
print(f"Erorr of svd: {error_svd}, error of mps: {error_mps}")
print(f"Compression factor of svd: {factor_svd}, compression factor of mps: {factor}")
print(f"Error of svd divided by compression factor: {error_svd / factor_svd}, error of mps divided by compression factor: {error_mps / factor}")

#Why is the MPS better than SVD
#The first aspect is time complexity, where svd has a time complexity of O(n^3) and mps has a time complexity of O(n*sqrt(n)*f(n))
#which makes MPS significantly faster than SVD for large n.
#the second aspect is that MPS has lower error with the same number of parameters

#plot difference of original image and svd and original image and mps
plt.subplot(1, 2, 1)
plt.title("SVD")
plt.imshow(image - rest, cmap='gray')
plt.subplot(1, 2, 2)
plt.title("MPS")
plt.imshow(image - t0prime, cmap='gray')
plt.show()

