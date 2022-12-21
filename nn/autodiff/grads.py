import numpy as np


def add_grad(a, b, dout): return dout, dout

def subtract_grad(a, b, dout): return dout, np.negative(dout) 

def multiply_grad(a, b, dout): 
	return np.multiply(dout, b.value), np.multiply(dout, a.value)

def negative_grad(a, dout): return np.negative(dout), dout

def divide_grad(a, b, dout):
	return (np.divide(dout, b.value),
	-np.multiply(dout, np.divide(a.value, np.power(b.value, 2))))

def power_grad(a, b, dout):
	return (np.multiply(dout, np.multiply(b.value, np.power(a.value, b.value - 1))),
	np.multiply(dout, np.multiply(np.log(a.value), np.power(a.value, b.value))))

def sin_grad(a, dout): return np.multiply(dout, np.cos(a.value)), dout
def cos_grad(a, dout): return np.multiply(dout, -np.sin(a.value)), dout
def tan_grad(a, dout): 
	return np.multiply(dout, np.add(1, np.power(np.tan(a.value), 2))), dout

def tanh_grad(a, dout): return np.multiply(dout, 1 - np.power(np.tanh(a.value), 2)), dout

def matmul_grad(a, b, dout):
	return np.matmul(dout, b.value.T), np.matmul(a.value.T, dout)

def transpose_grad(a, dout): return dout.T, dout.T