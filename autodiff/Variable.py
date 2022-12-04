import numpy as np
from numbers import Number
from autodiff.core import *


class Variable(np.lib.mixins.NDArrayOperatorsMixin):
	def __init__(self, value, dtype="float32", name=None, **kwargs) -> None:
		self.dtype = dtype
		self._i: np.ndarray = np.array(value, self.dtype)
		self.shape = self._i.shape
		self.name: str = name
		self._source: Node = kwargs.get("source") or Var(self._i, name=self.name)

	def __repr__(self) -> str:
		return f"<{Variable.__name__} value={self._i} dtype={self.dtype}>"

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

			# print(ufunc.__name__)
			# print(inputs)

			val = ufunc(*scalars, **kwargs)
			source = OPS.get(ufunc.__name__)(*inputs, value=val, name=self.name)

			return self.__class__(val, source=source, dtype=self.dtype)
		
	def __array_function__(self, func, types, args, kwargs):
		if func not in OPS:
			return NotImplemented

		if not all(issubclass(t, self.__class__) for t in types):
			return NotImplemented

		return OPS[func](*args, **kwargs)

	def numpy(self): return np.asarray(self)


def implements(np_function):
	def decorator(func):
		OPS[np_function] = func
		return func
	return decorator


@implements(np.transpose)
def T(a): 
	val = a._i.T
	source = OPS.get(np.transpose.__name__)(a._source, value=val) # TODO: add a operation type for transpose...
	return Variable(val, source=source, dtype=a.dtype)
