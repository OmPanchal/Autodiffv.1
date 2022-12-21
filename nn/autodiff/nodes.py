import numpy as np
from nn.autodiff.utils import set_name


class Node(object):
	def __init__(self, value, name) -> None:
		self.value = value
		self.gradient = 0
		self.name = name
		# print(Node.__subclasses__())


class Graph(object):
	_g = None

	def __init__(self) -> None:
		self.nodes = set()
		Graph._g = self.nodes

	def __reset_counts(self, root):
		if hasattr(root, "count"):
			root.count = 0
		else: 
			for child in root.__subclasses__():
				self.__reset_counts(child)

	def __reset_session(self):
		try:
			del Graph._g
		except: pass

		Graph._g = None
		self.__reset_counts(Node)
		self.__reset_counts(Operator)

	def __enter__(self):
		return self

	def __exit__(self, exc_type, exc_value, traceback):
		self.__reset_session()


class Var(Node):
	count = 0

	def __init__(self, value, name=None) -> None:
		self.name = set_name(Var) if name is None else name
		Var.count += 1
		# print("Var", Var.count)
		if Graph._g: Graph._g.add(self)
		super().__init__(value, name)
	
	def __repr__(self) -> str:
		return f"<{Var.__name__} name={self.name} value={self.value}>"


class Const(Node):
	count = 0

	def __init__(self, value, name=None) -> None:
		self.name = set_name(Const) if name is None else name
		if Graph._g: Graph._g.add(self)
		Const.count += 1
		super().__init__(value, name)

	def __repr__(self) -> str:
		return f"{Const.__name__} name={self.name} value={self.value}"


class Operator(Node):

	def __init__(self, inputs, value=None, name="Operator") -> None:
		self.inputs = inputs
		if Graph._g: Graph._g.add(self)
		self.grad = None
		super().__init__(value, name)

	def backward(self, dout): 
		# reset the gradient
		self.gradient = 0
		return self.grad(*self.inputs, dout)

	def __repr__(self) -> str:
		return f"{Operator.__name__} name={self.name} inputs={self.inputs}"