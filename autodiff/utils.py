from autodiff.core import Operator
from autodiff.Variable import Variable
import numpy as np


class Identity(object):
	def __init__(self, shape) -> None:
		self.shape = shape

	def __matmul__(self, other):
		result_shape = (self.shape[0], other.shape[1])
		print(result_shape)


def __g(dy, dx):
	op = [dy._source]
	dy._source.gradient = np.ones(dy._i.shape)
	# print(dy._source.gradient)

	for x in dx:
		x._source.gradient = 0 
	# ^ Reset the gradients for the variables involved ...

	for operation in op:
		grads = operation.backward(dout=operation.gradient)

		for inp, grad in zip(operation.inputs, grads):
			if inp: inp.gradient += grad

			if isinstance(inp, Operator):
				op.append(inp)

	for value in dx:
		yield value._source.gradient

def grad(dy, dx): return Variable(list(__g(dy, dx)), dtype=dy.dtype)
