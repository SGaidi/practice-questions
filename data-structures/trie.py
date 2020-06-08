import heapq
import random
import typing
import pytest

from typing import List
from collections import namedtuple, defaultdict


# Utilites


def random_string(min_length: int = 0, max_length: int = 10) -> str:
	letters = [chr(value) for value in range(ord('a'), ord('z')+1)]
	return "".join(random.choice(letters) for i in range(random.randint(min_length, max_length)))


def random_numbers(count: int = 10) -> List[int]:
	return [random.randint(0, 1) for _ in range(count)]


def random_matrix(rows: int = 10, cols: int = 10) -> List[List[int]]:
	return [random_numbers(cols) for _ in range(rows)]


# Question 1


class CharTrieNode1:

	CHAR_COUNT = ord('Z') - ord('A') + 1

	def __init__(self):
		self.children = [None for char in range(CharTrieNode1.CHAR_COUNT)]
		self.end = False
	
	def getChar(self, c: str):
		return self.children[ord(c)-ord('A')]
	
	def setChar(self, c: str):
		self.children[ord(c)-ord('A')] = CharTrieNode1()
	
	def removeChar(self, c: str):
		self.children[ord(c)-ord('A')] = None
	
	def __str__(self):
		stack = [self]
		chars = []
		while stack:
			curr = stack.pop()
			for idx, child in enumerate(curr.children):
				if child is not None:
					chars.append(chr(ord('A') + idx))
					stack.append(child)
		return ', '.join(c for c in chars)
	
	@staticmethod
	def validateWord(word: str):
		if word == "": raise ValueError
		if any(ord(c) < ord('A') or ord(c) > ord('Z') for c in word.upper()): raise ValueError
	
	def insert(self, word: str) -> None:
		curr = self
		for word_idx in range(len(word)):
			if curr.getChar(word[word_idx]) is None:
				curr.setChar(word[word_idx])
			curr = curr.getChar(word[word_idx])
			word_idx += 1
		curr.end = True
	
	def search(self, word: str) -> bool:
		curr = self
		for word_idx in range(len(word)):
			if curr.getChar(word[word_idx]) is None:
				return False
			curr = curr.getChar(word[word_idx])
			word_idx += 1
		return curr.end
	
	def remove(self, word: str) -> bool:
		if not self.search(word): raise ValueError
		
		if not word:
			return self.end
		else:
			curr_char = self.getChar(word[0])
			is_deleted = curr_char.remove(word[1:])
			if not any(c for c in curr_char.children if c is not None) and is_deleted:
				self.removeChar(word[0])
				return True
			else:
				return False


def test_trie1() -> None:
	trie = CharTrieNode1()
	trie.insert("A")
	assert str(trie) == "A"
	trie.insert("A")
	assert str(trie) == "A"
	trie.insert("B")
	assert str(trie) == "A, B"
	trie.insert("BB")
	assert str(trie) == "A, B, B"
	trie.insert("ABC")
	assert str(trie) == "A, B, B, B, C"
	trie.insert("A")
	assert str(trie) == "A, B, B, B, C"
	trie.insert("ADBC")
	assert str(trie) == "A, B, B, B, D, B, C, C"
	with pytest.raises(ValueError):
		trie.remove("AB")
	trie.remove("BB")
	assert str(trie) == "A, B, D, B, C, C"


# Question 2


class CharTrieNode2:

	END = "#"

	def __init__(self):
		self.children = {}
	
	def __str__(self):
		stack = [self]
		chars = []
		while stack:
			curr = stack.pop()
			for key, child in enumerate(curr.children.items()):
				if child is not None:
					chars.append(key)
					stack.append(child)
		return ', '.join(c for c in chars)
	
	def insert(self, word: str):
		if word == "":
			self.children[CharTrieNode2.END] = None
		else:
			if word[0] not in self.children:
				self.children[word[0]] = CharTrieNode2()
			self.children[word[0]].insert(word[1:])
	
	def search(self, word: str) -> bool:
		if word == "":
			return "#" in self.children
		else:
			return word[0] in self.children and self.children[word[0]].search(word[1:])
	
	def delete(self, word: str) -> bool:
		if not self.search(word): raise ValueError
		if word == "":
			del self.children["#"]
			return not self.chldren.keys()
		elif self.children.delete(word[1:]) and len(self.chldren.keys()) == 1:
			del self.children[word[0]]
			return True
		else:
			return False


def test_trie2() -> None:
	trie = CharTrieNode1()
	trie.insert("A")
	assert str(trie) == "A"
	trie.insert("A")
	assert str(trie) == "A"
	trie.insert("B")
	assert str(trie) == "A, B"
	trie.insert("BB")
	assert str(trie) == "A, B, B"
	trie.insert("ABC")
	assert str(trie) == "A, B, B, B, C"
	trie.insert("A")
	assert str(trie) == "A, B, B, B, C"
	trie.insert("ADBC")
	assert str(trie) == "A, B, B, B, D, B, C, C"
	with pytest.raises(ValueError):
		trie.remove("AB")
	trie.remove("BB")
	assert str(trie) == "A, B, D, B, C, C"


# Question 4


def lcp(strs: List[str]) -> int:
	"""returns length of longest commim prefix"""
	trie = CharTrieNode2()
	for string in strs:
		trie.insert(string)
	counter = 0
	while sum(1 for child in trie.children.values() if child != None) == 1:
		counter += 1
		print(list(key for key, child in trie.children.items() if child != None)[0])
		trie = list(child for child in trie.children.values() if child != None)[0]
	return counter


def test_lcp() -> None:
	assert 3 == lcp(["codeable", "code", "coder", "coding"])


# Question 5


def lexicalSorted(strings: List[str]) -> List[str]:
	trie = CharTrieNode2()
	for string in strings:
		trie.insert(string)
	StrNode = namedtuple('StrNode', ['trie', 'string'])
	strings_sorted = []
	stack = [StrNode(trie, "")]
	while stack:
		curr, string = stack.pop()
		for key, child in sorted(curr.children.items()):
			if child is not None:
				stack.append(StrNode(child, string+key))
			else:
				strings_sorted.append(string)
	return strings_sorted


def test_lexicalSorted() -> None:
	strings = [random_string() for _ in range(3)]
	assert sorted(strings)[::-1] == lexicalSorted(strings)


# Question 6


class CharTrieNode3:

	END = "#"

	def __init__(self):
		self.children = defaultdict(int)
	
	def __str__(self):
		stack = [self]
		chars = []
		while stack:
			curr = stack.pop()
			for key, child in curr.children.items():
				if key != '#':
					stack.append(child)
				chars.append((key, child))
		return ', '.join("{}:{}".format(c, count) for c, count in chars)
	
	def insert(self, word: str):
		if word == "":
			self.children[CharTrieNode3.END] += 1
		else:
			if word[0] not in self.children:
				self.children[word[0]] = CharTrieNode3()
			self.children[word[0]].insert(word[1:])
	
	def search(self, word: str) -> bool:
		if word == "":
			return "#" in self.children
		else:
			return word[0] in self.children and self.children[word[0]].search(word[1:])
	
	def delete(self, word: str) -> bool:
		if not self.search(word): raise ValueError
		if word == "":
			del self.children["#"]
			return not self.chldren.keys()
		elif self.children.delete(word[1:]) and len(self.chldren.keys()) == 1:
			del self.children[word[0]]
			return True
		else:
			return False
	
	def count(self, word: str) -> int:
		if word == "":
			return self.children['#']
		else:
			return self.children[word[0]].count(word[1:])


def max_counter1(strings: List[str]) -> str:
	if not strings: raise ValueError
	trie = CharTrieNode3()
	for string in strings:
		trie.insert(string)
	return max(((string, trie.count(string)) for string in strings), key=lambda t: t[1])[0]


def max_counter2(strings: List[str]) -> str:
	if not strings: raise ValueError
	trie = CharTrieNode3()
	for string in strings:
		trie.insert(string)
	StrNode = namedtuple('StrNode', ['trie', 'string'])
	stack = [StrNode(trie, "")]
	max_count = 0; max_string = ""

	while stack:
		curr, string = stack.pop()
		for key, child in curr.children.items():
			if key == '#':
				if child > max_count:
					max_count = child
					max_string  = string
			else:
				stack.append(StrNode(child, string+key))
	return max_string


def simple_max_counter(strings: List[str]) -> str:
	counter = defaultdict(int)
	for string in strings:
		counter[string] += 1
	return max(((string, count) for string, count in counter.items()), key=lambda t: t[1])[0]

def test_max_counter() -> None:
	strings = [random_string(min_length=1, max_length=2) for _ in range(200)]
	assert max_counter1(strings) == simple_max_counter(strings) == max_counter2(strings)


# Question 7


def max_n(strings: List[str], n: int) -> List[str]:
	if not strings: raise ValueError
	if n < 0: raise ValueError
	trie = CharTrieNode3()
	for string in strings:
		trie.insert(string)
	StrNode = namedtuple('StrNode', ['trie', 'string'])
	stack = [StrNode(trie, "")]
	heap = []
	while stack:
		curr, string = stack.pop()
		for key, child in curr.children.items():
			if key == '#':
				heapq.heappush(heap, (child, string))
			else:
				stack.append(StrNode(child, string+key))
	return [item[1] for item in heapq.nlargest(n, heap)]


def test_max_n() -> None:
	strings = [random_string(min_length=1, max_length=2) for _ in range(200)]
	n = random.randint(0, 10)
	print(max_n(strings, n))


# Question 8


def duplicate_rows_in(matrix: List[List[int]]) -> List[List[int]]:
	trie = CharTrieNode3()
	for row in matrix:
		trie.insert(''.join(str(num) for num in row))
	StrNode = namedtuple('StrNode', ['trie', 'string'])
	stack = [StrNode(trie, "")]
	duplicates = []
	while stack:
		curr, string = stack.pop()
		for key, child in curr.children.items():
			if key == '#':
				if child > 1:
					duplicates.append(string)
			else:
				stack.append(StrNode(child, string+key))
	return duplicates


def test_duplicate_rows_in() -> None:
	matrix = random_matrix(10, 3)
	for row in matrix:
		print(row)
	print()
	print(duplicate_rows_in(matrix))


# Question 9


def break_problem(sentence: str, words: List[str]) -> bool:
	if not sentence: return True
	previous = [False for _ in range(len(sentence))]
	trie = CharTrieNode3()
	max_word = len(max(words, key=lambda word: len(word)))
	for word in words:
		trie.insert(word)
	for end_idx in range(len(sentence)):
		# first naive option - one word from 0 to end_idx+1
		if trie.search(sentence[:end_idx+1]):
			previous[end_idx] = True
		# second option - previous breaks
		for prev_idx in range(min(end_idx, max_word), -1, -1):
			if previous[prev_idx] and trie.search(sentence[prev_idx+1:end_idx+1]):
				previous[prev_idx] = True
	return previous[-1]


def test_break_problem() -> None:
	string = random_string(min_length=1, max_length=3)
	words = [random_string(min_length=1, max_length=2) for _ in range(random.randint(1, 30))]
	print(string)
	print(words)
	print(break_problem(string, words))
