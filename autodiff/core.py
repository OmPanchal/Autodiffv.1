import numpy as np


# TODO: Add Handling for other arr functions later

#?       ___\/___
#?     /         \
#?    |    .   . |
#?   | V    v   |
#?   \_________/
#?     |   |


class Node(object):
	def __init__(self, value, name) -> None:
		self.value: Node = value
		self.gradient: np.ndarray = 0
		self.name: str = name


class Var(Node):
	count: int = 0

	def __init__(self, value, name=None) -> None:
		self.name: str = f"Var/{Var.count}" if name is None else name
		Var.count+= 1
		super().__init__(value, self.name)

	def __repr__(self) -> str:
		return f"<Var name:{self.name} value:{self.value}>"


class Operator(Node):
	count: int = 0

	def __init__(self, value=None, name="Operator") -> None:
		self.inputs = []
		super().__init__(value, name)

	def __repr__(self) -> str:
		return f"<Operator name:{self.name} inputs:{self.inputs}>"

# ADD/SUB
class add(Operator):
	count: int = 0

	def __init__(self, a, b, value=None, name=None) -> None:
		super().__init__(value, name)
		self.inputs: list = [a, b]
		self.name: str = f"add/{add.count}" if name is None else name
		add.count += 1

	def backward(self, dout):
		return dout, dout
class subtract(Operator):
	count: int = 0

	def __init__(self, a, b, value=None, name=None) -> None:
		super().__init__(value, name)
		self.inputs: list = [a, b]
		self.name: str = f"sub/{subtract.count}" if name is None else name
		subtract.count += 1

	def backward(self, dout):
		return dout, np.negative(dout)


# MUL/NEG/DIV
class multiply(Operator):
	count: int = 0

	def __init__(self, a, b, value=None, name=None) -> None:
		super().__init__(value, name)
		self.inputs: list = [a, b]
		self.name: str = f"multiply/{multiply.count}" if name is None else name
		multiply.count += 1

	@classmethod
	def __back(cls, a, b, dout):
		return np.multiply(dout, b.value), np.multiply(dout, a.value)
	def backward(self, dout):
		return multiply.__back(*self.inputs, dout)
class negative(Operator):
	count: int = 0

	def __init__(self, a, b=None, value=None, name=None) -> None:
		super().__init__(value, name)
		self.inputs: list = [a]
		self.name: str = f"negative/{negative.count}" if name is None else name
		negative.count += 1

	@classmethod
	def __back(cls, a, dout):
		return np.negative(dout), dout
	def backward(self, dout):
		return negative.__back(*self.inputs, dout)


class divide(Operator):
	count: int = 0

	def __init__(self, a, b, value=None, name=None) -> None:
		super().__init__(value, name)
		self.inputs: list = [a, b]
		self.name: str = f"divide/{divide.count}" if name is None else name
		divide.count += 1

	@classmethod
	def __back(cls, a, b, dout):
		return (np.divide(dout, b.value), 
		-np.multiply(dout, np.divide(a.value, np.power(b.value, 2))))
	def backward(self, dout):
		return divide.__back(*self.inputs, dout)


# POWER
class power(Operator):
	count: int = 0

	def __init__(self, a, b, value=None, name=None) -> None:
		super().__init__(value, name)
		self.inputs = [a, b]
		self.name = f"power/{power.count}" if name is None else name
		power.count += 1

	@classmethod
	def __back(cls, a, b, dout):
		return (np.multiply(dout, np.multiply(b.value, np.power(a.value, b.value - 1))), 
				np.multiply(dout, np.multiply(np.log(a.value), np.power(a.value, b.value))))
	def backward(self, dout):
		return power.__back(*self.inputs, dout)



# TRIG
class sin(Operator):
	count: int = 0
	
	def __init__(self, a, b=None, value=None, name=None) -> None:
		super().__init__(value, name)
		self.inputs = [a]
		self.name = f"sin/{sin.count}" if name is None else name
		sin.count += 1

	@classmethod
	def __back(cls, a, dout):
		return np.multiply(dout, np.cos(a.value)), dout
	def backward(self, dout):
		return sin.__back(*self.inputs, dout)
class cos(Operator):
	count: int = 0
	
	def __init__(self, a, b=None, value=None, name=None) -> None:
		super().__init__(value, name)
		self.inputs = [a]
		self.name = f"cos/{cos.count}" if name is None else name
		cos.count += 1

	@classmethod
	def __back(cls, a, dout):
		return np.multiply(dout, -np.sin(a.value)), dout
	def backward(self, dout):
		return cos.__back(*self.inputs, dout)
class tan(Operator):
	count: int = 0
	
	def __init__(self, a, b=None, value=None, name=None) -> None:
		super().__init__(value, name)
		self.inputs = [a]
		self.name = f"tan/{tan.count}" if name is None else name
		tan.count += 1

	@classmethod
	def __back(cls, a, dout):
		return (np.multiply(dout, np.add(1, np.power(np.tan(a.value), 2))),
		 dout)
	def backward(self, dout):
		return tan.__back(*self.inputs, dout)


# BIG BOY MATMUL

class matmul(Operator) :
	count: int = 1

	def __init__(self, a, b, value=None, name=None) -> None:
		super().__init__(value, name)
		self.inputs = [a, b]
		self.name = f"matmul/{matmul.count}" if name is None else name
		matmul.count += 1

	@classmethod
	def __back(cls, a, b, dout):
		print("NICE", np.matmul(dout, b.value.T).shape)
		return np.matmul(dout, b.value.T), np.matmul(a.value.T, dout)
	def backward(self, dout):
		return matmul.__back(*self.inputs, dout)


class transpose(Operator):
	count: int = 0

	def __init__(self, a, b=None, value=None, name=None) -> None:
		super().__init__(value, name)
		self.inputs = [a]
		self.name = f"transpose/{transpose.count}" if name is None else name
		transpose.count += 1

	@classmethod
	def __back(cls, a, dout):
		return dout.T, dout.T
	def backward(self, dout):
		return self.__back(*self.inputs, dout)


OPS = {
	np.add.__name__: add,
	np.subtract.__name__: subtract,
	np.multiply.__name__: multiply,
	np.divide.__name__: divide,
	np.negative.__name__: negative,
	np.power.__name__: power,
	# TRIG
	np.sin.__name__: sin,
	np.cos.__name__: cos,
	np.tan.__name__: tan,
	# LINLAG...
	np.matmul.__name__: matmul,
	np.transpose.__name__: transpose
	}
