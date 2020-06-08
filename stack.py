import typing
import pytest


# Question 2


class Node:
	
	def __init__(self, data, next = None):
		self.data = data
		self.next = next
	
	def __str__(self):
		return "Node: data={}, next.data={}".format(self.data, self.next.data if self.next is not None else "None")

class Stack:
	
	def __init__(self):
		self.head = None
	
	class EmptyException(Exception):
		
		def __str__(self):
			return "Stack is empty!"
	
	def is_empty(self) -> bool:
		return self.head is None
	
	def push(self, data) -> None:
		self.head = Node(data, self.head)
	
	def pop(self) -> Node:
		if self.is_empty():
			raise Stack.EmptyException()
		else:
			data = self.head.data
			self.head = self.head.next
			return data
	
	def peek(self) -> Node:
		if self.is_empty():
			raise Stack.EmptyException()
		else:
			return self.head.data
	

def test_Stack():
	stack = Stack()
	stack.push(1)
	stack.push(3)
	stack.push(2)
	assert stack.pop() == 2
	assert stack.peek() == 3
	assert not stack.is_empty()
	assert stack.pop() == 3
	stack.push(4)
	assert stack.pop() == 4
	assert stack.pop() == 1
	assert stack.is_empty()
	with pytest.raises(Stack.EmptyException):
		stack.pop()
	with pytest.raises(Stack.EmptyException):
		stack.peek()


test_Stack()