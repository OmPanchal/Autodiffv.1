import nn.autodiff as ad
import numpy as np
import matplotlib.pyplot as plt
import time as time


#! WORKING NEURAL NETWORK

W1 = ad.Variable(np.random.rand(200, 2), dtype="float64")
B1 = ad.Variable(np.random.rand(200, 1), dtype="float64")

W2 = ad.Variable(np.random.rand(1, 200), dtype="float64")
B2 = ad.Variable(np.random.rand(1, 1), dtype="float64")

C2 = ad.Variable(np.random.rand(1, 1))

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])[..., np.newaxis]
Y = np.array([0, 1, 1, 0])[..., np.newaxis]

PARAMS = np.array([W1, B1, W2, B2], dtype="object")

# print(X.shape, Y.shape)


s = time.time()

for i in range(1000):
	with ad.Graph() as g:
		for x, y in zip(X, Y):
			
			# A = ad.Variable(2)
			# print(A._source.count)
			# B = (A ** 2) - (4 * A) + 17

			Z1 = np.matmul(PARAMS[0], x) + PARAMS[1]
			A1 = np.tanh(Z1)
			Z2 = np.matmul(PARAMS[2], A1) + PARAMS[3]
			# print(Z2)
			LOSS = (y - Z2) ** 2

		GRAD = ad.grad(LOSS, [W1, B1, W2, B2])
		# print(GRAD)
		PARAMS -= 0.001 * GRAD


print(LOSS)
print((time.time() - s))

# for p, g in zip(PARAMS, GRAD):
# 	print("ORIGIONAL", p.shape, "GRAD", g.shape)