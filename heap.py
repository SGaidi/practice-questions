import pytest
import random
import itertools

from typing import List


# Questions 1-4


class BinaryMaxHeap:

	INVALID_INDEX = None

	def __init__(self, items: list = []):
		self.items = []
		for item in items:
			self.push(item)
	
	def __str__(self):
		return "BinaryMaxHeap: [{}]".format(", ".join(str(item) for item in self.items))
	
	class HeapEmpty(Exception):
		pass
	
	def size(self) -> int:
		return len(self.items)
	
	def isEmpty(self) -> bool:
		return self.size() == 0
	
	def top(self):
		if self.isEmpty():
			raise BinaryMaxHeap.HeapEmpty()
		return self.items[0]
	
	def parentOf(self, idx: int) -> int:
		parent_idx = (idx - 1) // 2
		return parent_idx if parent_idx >= 0 else BinaryMaxHeap.INVALID_INDEX
	
	def leftChildOf(self, idx: int) -> int:
		left_idx = idx * 2 + 1
		return left_idx if left_idx < self.size() else BinaryMaxHeap.INVALID_INDEX
	
	def rightChildOf(self, idx: int) -> int:
		right_idx = idx * 2 + 2
		return right_idx if right_idx < self.size() else BinaryMaxHeap.INVALID_INDEX
	
	def heapifyUp(self) -> None:
		curr_idx = len(self.items) - 1
		parent_idx = self.parentOf(curr_idx)
		
		while curr_idx is not BinaryMaxHeap.INVALID_INDEX and parent_idx is not BinaryMaxHeap.INVALID_INDEX:
			if self.items[curr_idx] < self.items[parent_idx]:
				break
			
			self.items[curr_idx], self.items[parent_idx] = self.items[parent_idx], self.items[curr_idx]
			
			curr_idx = parent_idx
			parent_idx = self.parentOf(curr_idx)
	
	def push(self, item) -> None:
		self.items.append(item)
		self.heapifyUp()
	
	def heapifyDown(self, idx: int = 0) -> None:
		curr_idx = idx
		left_idx, right_idx = self.leftChildOf(curr_idx), self.rightChildOf(curr_idx)
		
		while True:
			max_idx = curr_idx
			if left_idx is BinaryMaxHeap.INVALID_INDEX:
				break
			if self.items[curr_idx] < self.items[left_idx]:
				max_idx = left_idx
			if right_idx is not BinaryMaxHeap.INVALID_INDEX and self.items[max_idx] < self.items[right_idx]:
				max_idx = right_idx
			
			if max_idx == curr_idx:
				break
			self.items[curr_idx], self.items[max_idx] = self.items[max_idx], self.items[curr_idx]

			curr_idx = max_idx
			left_idx, right_idx = self.leftChildOf(curr_idx), self.rightChildOf(curr_idx)
	
	def pop(self):
		item = self.top()
		if self.size() > 1:
			self.items[0] = self.items.pop()
			self.heapifyDown()
		else:
			self.items.pop()
		return item
	
	# Question 5
	
	@staticmethod
	def heapsorted(items) -> List:
		heap = BinaryMaxHeap()
		for item in items:
			heap.push(item)
		sorted_items = []
		while not heap.isEmpty():
			sorted_items.append(heap.pop())
		return sorted_items
	
	# Question 6
	
	@classmethod
	def isBinaryMaxHeap(cls, items) -> bool:
		heap = BinaryMaxHeap()
		heap.items = items
		
		for parent_idx in range(heap.size() // 2):
			max_val = heap.items[parent_idx]
			left_idx, right_idx = heap.leftChildOf(parent_idx), heap.rightChildOf(parent_idx)

			if left_idx is not cls.INVALID_INDEX and heap.items[left_idx] > max_val:
					return False
			if right_idx is not cls.INVALID_INDEX and heap.items[right_idx] > max_val:
					return False
		
		return True
	
	# Question 7
	## Version 1
	
	@classmethod
	def getKSmallest(cls, items, k: int) -> List:
		if len(items) <= k: return items
		
		heap = cls()
		for idx in range(k):
			heap.push(items[idx])
		for idx in range(k, len(items)):
			if heap.top() > items[idx]:
				heap.items[0] = items[idx]
				heap.heapifyDown()
		
		return heap.items
	
	## Version 2
	
	@classmethod
	def getKLargest(cls, items, k: int) -> List:
		if len(items) <= k: return items
		
		heap = cls(items)
		largest = []
		for _ in range(k):
			largest.append(heap.pop())
		return largest
	
	# Question 8
	
	@classmethod
	def sortedKSorted(cls, items, k: int) -> List:
		heap = cls()
		sorted_items = []
		for idx in range(k+1):
			heap.push(items[-idx])
		for idx in range(k+1, len(items)):
			if heap.top() > items[-idx]:
				sorted_items.append(heap.pop())
				heap.push(items[-idx])
			else:
				sorted_items.append(items[-idx])
		while not heap.isEmpty():
			sorted_items.append(heap.pop())
		return sorted_items

	# Question 9

	@classmethod
	def mergeSorted(cls, sorted_lists: List[List]) -> List:
		heap = cls()
		merged = []
		
		for idx, sorted_list in enumerate(sorted_lists):
			if sorted_lists[idx]:
				heap.push((sorted_list.pop(), idx))
		
		while not all(not sorted_list for sorted_list in sorted_lists):
			value, idx = heap.pop()
			merged.append(value)
			if sorted_lists[idx]:
				heap.push((sorted_lists[idx].pop(), idx))
		
		while not heap.isEmpty():
			merged.append(heap.pop()[0])
		
		return merged


@pytest.fixture
def binary_heap() -> BinaryMaxHeap:
	return BinaryMaxHeap()


def generate_random_ints(min_val: int, max_val: int, min_len: int, max_len: int) -> List[int]:
	return [random.randint(min_val, max_val+1) for count in range(random.randint(min_len, max_len+1))]


@pytest.fixture
def random_ints() -> List[int]:
	return generate_random_ints(-20, 20, 0, 20)


@pytest.fixture
def random_sorted_lists() -> List[List[int]]:
	return [sorted(generate_random_ints(-20, 20, 0, 20)) for count in range(random.randint(0, 10))]


def push_ints(binary_heap: BinaryMaxHeap, random_ints: List[int]):
	for rand_int in random_ints:
		binary_heap.push(rand_int)


def test_push(binary_heap: BinaryMaxHeap, random_ints: List[int]):
	push_ints(binary_heap, random_ints)


def test_size(binary_heap: BinaryMaxHeap, random_ints: List[int]):
	assert binary_heap.size() == 0, "heap should be empty, but it's size is {}".format(binary_heap.size())
	push_ints(binary_heap, random_ints)
	assert len(random_ints) == binary_heap.size(), \
		"expected heap to be of size {}, but is actually {}".format(
			len(random_ints), binary_heap.size())


def test_pop(binary_heap: BinaryMaxHeap, random_ints: List[int]):
	push_ints(binary_heap, random_ints)
	popped_items = []
	while binary_heap.size():
		popped_items.append(binary_heap.pop())
	for rand_int in random_ints:
		assert rand_int in popped_items, "item {} is not in popped items".format(rand_int)
	assert popped_items == sorted(random_ints, reverse=True), \
		"popped items are not decreasingly sorted: {}, should be: {}".format(popped_items, sorted(random_ints, reverse=True))


def test_heapsort(random_ints: List[int]):
	sorted_ints = sorted(random_ints, reverse=True)
	heapsroted_ints = BinaryMaxHeap.heapsorted(random_ints)
	assert heapsroted_ints == sorted_ints, "heapsort resulted in: {}, expected: {}".format(
		heapsroted_ints, sorted_ints)


def test_isBinaryMaxHeap(binary_heap: BinaryMaxHeap, random_ints: List[int]):
	assert not BinaryMaxHeap.isBinaryMaxHeap(random_ints), "random ints {} is unlikely to be a max heap".format(random_ints)
	heap = BinaryMaxHeap(random_ints)
	assert BinaryMaxHeap.isBinaryMaxHeap(heap.items), "a heap created from list should be verified by isBinaryMaxHeap".format(heap.items)


def test_getKSmallest(random_ints: List[int], k: int = 4):
	k = min(k, len(random_ints)-1)
	k_smallest_from_heap = BinaryMaxHeap.getKSmallest(random_ints, k)
	assert len(k_smallest_from_heap) == k, "should extract {} items, but extracted {} items".format(
		k, len(k_smallest_from_heap))
	k_smallest_expected = sorted(random_ints)[:k]
	assert sorted(k_smallest_from_heap) == k_smallest_expected, "expected {} items to be: {}, but actually: {}".format(
		k, k_smallest_expected, k_smallest_from_heap)


def test_getKLargest(random_ints: List[int], k: int = 4):
	k = min(k, len(random_ints)-1)
	k_largest_from_heap = BinaryMaxHeap.getKLargest(random_ints, k)
	assert len(k_largest_from_heap) == k, "should extract {} items, but extracted {} items".format(
		k, len(k_largest_from_heap))
	k_largest_expected = sorted(random_ints)[-k:]
	assert sorted(k_largest_from_heap) == k_largest_expected, "expected {} items to be: {}, but actually: {}".format(
		k, k_largest_expected, k_largest_from_heap)


def test_sortedKSorted():
	items = [1, 4, 5, 2, 3, 7, 8, 6, 10, 9]
	k = 2
	sorted_by_heap = BinaryMaxHeap.sortedKSorted(items, k)
	assert sorted_by_heap == sorted(items, reverse=True), "expected {} to be sorted, but actually: {}".format(items, sorted_by_heap)


def test_mergedSorted(random_sorted_lists: List[List[int]]):
	expected_merge = sorted(itertools.chain(*random_sorted_lists), reverse=True)
	actual_merge = BinaryMaxHeap.mergeSorted(random_sorted_lists)
	assert expected_merge == actual_merge, \
		"expected {}, but resulted: {}".format(expected_merge, actual_merge)