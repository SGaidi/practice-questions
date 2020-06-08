import typing
import pytest


class BinarySearchTree:

	class Node:
		
		def __init__(self, key, value, left=None, right=None):
			self.key = key
			self.value = value
			self.left = left
			self.right = right
			
	class KeyAlreadyExists(Exception):
		"""raised when trying to insert already existing key"""
	
	class KeyNotExists(Exception):
		"""raised when searching for non-existing key"""
	
	def __init__(self):
		self.head = None
	
	def isEmpty(self) -> bool:
		return self.head is None
	
	# Question 1
	
	def insert(self, key, value) -> None:
		if self.isEmpty():
			self.head = BinarySearchTree.Node(key, value)
		else:
			curr = self.head
			while True:
				if curr.key == key:
					raise BinarySearchTree.KeyAlreadyExists()
				elif curr.key < key:
					if curr.left is None:
						curr.left = BinarySearchTree.Node(key, value)
						break
					else:
						curr = curr.left
				else: # curr.key > key
					if curr.right is None:
						curr.right = BinarySearchTree.Node(key, value)
						break
					else:
						curr = curr.right
	
	# Question 2
	
	def search(self, key):
		if self.isEmpty():
			raise BinarySearchTree.KeyNotExists()
		curr = self.head
		while True:
			if curr.key == key:
				return curr.value
			elif curr.key < key:
				if curr.left is None:
					raise BinarySearchTree.KeyNotExists()
				else:
					curr = curr.left
			else: # curr.key > key
				if curr.right is None:
					raise BinarySearchTree.KeyNotExists()
				else:
					curr = curr.right


def test_BinarySearchTree() -> None:
	tree = BinarySearchTree()
	assert tree.isEmpty()
	tree.insert(5, 4)
	assert not tree.isEmpty()
	assert tree.search(5) == 4
	tree.insert(3, 2)
	tree.insert(6, 1)
	assert tree.search(5) == 4
	assert tree.search(3) == 2
	assert tree.search(6) == 1
	with pytest.raises(BinarySearchTree.KeyAlreadyExists):
		tree.insert(5, 0)
	with pytest.raises(BinarySearchTree.KeyAlreadyExists):
		tree.insert(3, 7)
	with pytest.raises(BinarySearchTree.KeyAlreadyExists):
		tree.insert(6, 1)
	with pytest.raises(BinarySearchTree.KeyNotExists):
		tree.search(7)
	with pytest.raises(BinarySearchTree.KeyNotExists):
		tree.search(4)


test_BinarySearchTree()