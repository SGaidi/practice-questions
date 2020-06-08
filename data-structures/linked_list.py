import typing


class Node:

	def __init__(self, data, next=None):
		if next is not None and not isinstance(next, Node):
			raise ValueError("Node __init__: illegal `next` argument type ({}), not Node".format(type(next).__name__))
		self.data = data
		self.next = next
		
	def __str__(self) -> str:
		return "Node: {}".format(eslf.data)


class LinkedList:

	def __init__(self):
		self.head = None
		
	def __str__(self) -> str:
		return '\n'.join(str(item) for item in self.items())
	
	def insert(self, item) -> None:
		self.head = Node(item, self.head)
	
	def items(self) -> []:
		items = []
		current = self.head
		while current is not None:
			items.append(current.data)
			current = current.next
		return items
Node(3)
l = LinkedList()
l.insert(3)
print(l)