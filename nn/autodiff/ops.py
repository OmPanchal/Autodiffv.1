from nn.autodiff.nodes import (Operator, Graph)
from nn.autodiff.utils import set_name
from nn.autodiff.grads import *


class add(Operator):
	count = 0

	def __init__(self, inputs, value=None, name=None) -> None:
		super().__init__(inputs, value, name)
		if Graph._g: Graph._g.add(self)
		self.grad = add_grad
		self.name = set_name(add)
		add.count += 1

class subtract(Operator):
	count = 0

	def __init__(self, inputs, value=None, name=None) -> None:
		super().__init__(inputs, value, name)
		if Graph._g: Graph._g.add(self)
		self.grad = subtract_grad
		self.name = set_name(subtract)
		subtract.count += 1

class multiply(Operator):
	count = 0

	def __init__(self, inputs, value=None, name=None) -> None:
		super().__init__(inputs, value, name)
		if Graph._g: Graph._g.add(self)
		self.grad = multiply_grad
		self.name = set_name(multiply)
		multiply.count += 1

class negative(Operator):
	count = 0

	def __init__(self, inputs, value=None, name=None) -> None:
		super().__init__(inputs, value, name)
		if Graph._g: Graph._g.add(self)
		self.grad = negative_grad
		self.name = set_name(negative)
		negative.count += 1

class divide(Operator):
	count = 0

	def __init__(self, inputs, value=None, name=None) -> None:
		super().__init__(inputs, value, name)
		if Graph._g: Graph._g.add(self)
		self.grad = divide_grad
		self.name = set_name(divide)
		divide.count += 1

class power(Operator):
	count = 0

	def __init__(self, inputs, value=None, name=None) -> None:
		super().__init__(inputs, value, name)
		if Graph._g: Graph._g.add(self)
		self.grad = power_grad
		self.name = set_name(power)
		power.count += 1

class sin(Operator):
	count = 0

	def __init__(self, inputs, value=None, name=None) -> None:
		super().__init__(inputs, value, name)
		if Graph._g: Graph._g.add(self)
		self.grad = sin_grad
		self.name = set_name(sin)
		sin.count += 1

class cos(Operator):
	count = 0

	def __init__(self, inputs, value=None, name=None) -> None:
		super().__init__(inputs, value, name)
		if Graph._g: Graph._g.add(self)
		self.grad = cos_grad
		self.name = set_name(cos)
		cos.count += 1

class tan(Operator):
	count = 0

	def __init__(self, inputs, value=None, name=None) -> None:
		super().__init__(inputs, value, name)
		if Graph._g: Graph._g.add(self)
		self.grad = tan_grad
		self.name = set_name(tan)
		tan.count += 1

class tanh(Operator):
	count = 0

	def __init__(self, inputs, value=None, name=None) -> None:
		super().__init__(inputs, value, name)
		if Graph._g: Graph._g.add(self)
		self.grad = tanh_grad 
		self.name = set_name(tanh)
		tanh.count += 1

class matmul(Operator):
	count = 0

	def __init__(self, inputs, value=None, name=None) -> None:
		super().__init__(inputs, value, name)
		if Graph._g: Graph._g.add(self)
		self.grad = matmul_grad
		self.name = set_name(matmul)
		# print("MATMUL", matmul.count)
		matmul.count += 1

class transpose(Operator):
	count = 0
	
	def __init__(self, inputs, value=None, name=None) -> None:
		super().__init__(inputs, value, name)
		if Graph._g: Graph._g.add(self)
		self.grad = transpose_grad
		self.name = set_name(transpose)
		transpose.count += 1


OPS = {
	np.add.__name__: add,
	np.subtract.__name__: subtract,
	np.multiply.__name__: multiply,
	np.divide.__name__: divide,
	np.negative.__name__: negative,
	np.power.__name__: power,
	np.sin.__name__: sin,
	np.cos.__name__: cos,
	np.tan.__name__: tan,
	np.tanh.__name__: tanh,
	np.matmul.__name__: matmul,
	np.transpose.__name__: transpose
	}