import numpy as np
from numbers import Number


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


class add(Operator):
	count: int = 0

	def __init__(self, a, b, value=None, name=None) -> None:
		super().__init__(value, name)
		self.inputs: list = [a, b]
		self.name: str = f"add/{add.count}" if name is None else name
		add.count += 1

	def backward(self, dout):
		return dout, dout
class sub(Operator):
	count: int = 0

	def __init__(self, a, b, value=None, name=None) -> None:
		super().__init__(value, name)
		self.inputs: list = [a, b]
		self.name: str = f"sub/{sub.count}" if name is None else name
		add.count += 1

	def backward(self, dout):
		return dout, np.negative(dout)


class multiply(Operator):
	count: int = 0

	def __init__(self, a, b, value=None, name=None) -> None:
		super().__init__(value, name)
		self.inputs: list = [a, b]
		self.name: str = f"multiply/{multiply.count}" if name is None else name
		multiply.count += 1

	@classmethod
	def back(cls, a, b, dout):
		return np.multiply(dout, b.value), np.multiply(dout, a.value)
	def backward(self, dout):
		return multiply.back(*self.inputs, dout)
class negative(Operator):
	count: int = 0

	def __init__(self, a, b=Var(-1), value=None, name=None) -> None:
		super().__init__(value, name)
		self.inputs: list = [a, b]
		self.name: str = f"negative/{negative.count}" if name is None else name
		negative.count += 1

	@classmethod
	def back(cls, a, b, dout):
		return np.negative(dout), dout
	def backward(self, dout):
		return multiply.back(*self.inputs, dout)


class divide(Operator):
	count: int = 0

	def __init__(self, a, b, value=None, name=None) -> None:
		super().__init__(value, name)
		self.inputs: list = [a, b]
		self.name: str = f"divide/{divide.count}" if name is None else name
		divide.count += 1

	@classmethod
	def back(cls, a, b, dout):
		return (np.divide(dout, b.value), 
		-np.multiply(dout, np.divide(a.value, np.power(b.value, 2))))
	def backward(self, dout):
		return divide.back(*self.inputs, dout)

OPS = {
	"add": add,
	"multiply": multiply,
	"divide": divide,
	"subtract": sub,
	"negative": negative
	}


class Variable(np.lib.mixins.NDArrayOperatorsMixin):
	def __init__(self, value, dtype="float32", name=None, **kwargs) -> None:
		self._i: np.ndarray = np.array(value, dtype)
		self.name: str = name
		self._source: Node = kwargs.get("source") or Var(self._i, name=self.name)

	def __repr__(self) -> str:
		return f"<{Variable.__name__} value={self._i}>"

	def __array__(self):
		return self._i

	def __array_ufunc__(self, ufunc, method, *inps, **kwargs):
		if method == "__call__":
			scalars = []
			inputs = []

			for input in inps:
				if isinstance(input, Number):
					var = self.__class__(input)
					scalars.append(var._i)
					inputs.append(var._source) 
				
				elif isinstance(input, self.__class__): 
					scalars.append(input._i)
					inputs.append(input._source) 

				else: return NotImplemented

			print(ufunc.__name__)
			print(inputs)

			val = ufunc(*scalars, **kwargs)
			source = OPS.get(ufunc.__name__)(*inputs, value=val, name=self.name)

			return self.__class__(val, source=source)

	def numpy(self): return np.asarray(self)


def grad(dy, dx):
	op = [dy._source]
	dy._source.gradient = 1

	for operation in op:
		grads = operation.backward(dout=operation.gradient)

		for inp, grad in zip(operation.inputs, grads):
			# inp.gradient = 0 #! resetting the gradient...
			inp.gradient += grad

			if isinstance(inp, Operator):
				op.append(inp)

	for value in dx:
		yield value._source.gradient


if __name__ == "__main__":
	A = Variable([1, 2, 3])
	B = Variable([2, 3, 5])
	C = (-A * B) / (2 * A)

	print(C.__dict__)
	print(list(grad(C, [A, B])))
	