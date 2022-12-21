import numpy as np
from numbers import Number
from nn.autodiff.ops import OPS
from nn.autodiff.nodes import Operator, Var


class Variable(np.lib.mixins.NDArrayOperatorsMixin):
	def __init__(self, value, dtype="float32", name=None, **kwargs) -> None:
		self.dtype = dtype
		self._i = np.array(value, dtype=self.dtype)
		self.shape = self._i.shape
		self.size = self._i.size
		self.name = name
		self._source = kwargs.get("source") or Var(self._i, name=self.name)
		super().__init__()
		
	def __repr__(self) -> str:
		return f"<{Variable.__name__} value={self._i} dtype={self.dtype}>"
	
	def __getitem__(self, key):
		return self._i[key]

	def __array_ufunc__(self, ufunc, method, *inps, **kwargs):
		if method == "__call__":
			scalars = []
			sources = []
			for inp in inps:
				if isinstance(inp, Number) or isinstance(inp, np.ndarray):
					var = self.__class__(inp)
					scalars.append(var._i) # !!
					sources.append(var._source)

				elif isinstance(inp, self.__class__):
					scalars.append(inp._i)
					sources.append(inp._source)

				else: return NotImplemented

		val = ufunc(*scalars, **kwargs)
		source = OPS.get(ufunc.__name__)(sources, value=val)
		
		return self.__class__(val, source=source, dtype=self.dtype)

	def __array_function__(self, func, types, args, kwargs):
		if func not in OPS:
			return NotImplemented
		
		if not all(issubclass(t, self.__class__) for t in types):
			return NotImplemented

		return OPS.get(func)(*args, **kwargs)

	def numpy(self): return self._i


def implements(np_function):
	def decorator(func):
		OPS[np_function] = func
		return func
	return decorator


@implements(np.transpose)
def T(a):
	source = OPS.get(np.transpose.__name__)([a._source], value=a._i.T)
	return Variable(a._i.T, source=source, dtype=a.dtype)


def __g(dy, dx):
	op = [dy._source]
	dy._source.gradient = np.ones(dy.shape)
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

def grad(dy, dx): return np.array(list(__g(dy, dx)), dtype="object")