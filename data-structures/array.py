import random
from typing import List, Tuple

import pytest
import numpy as np


# Utilites


@pytest.fixture
def random_ints() -> List[int]:
	return [random.randint(-20, 20) for count in range(10)]


# Question 1 - pair of numbers with sum in unsroted array
## Version 1


def pair_with_sum1(numbers: List[int], sum: int) -> tuple:
	"""returns pair of numbers from `numbers` that their sum is `sum`,
	If no such pair exists, raises ValueError."""
	complements = set([])
	
	for num in numbers:
		if sum-num in complements:
			return (num, sum-num)
		else:
			complements.add(num)
	
	raise ValueError("pair_with_sum1: No pair with sum {} was found in numbers.".format(sum))
	
	
## Version 2


def pair_with_sum2(numbers: List[int], sum: int) -> List[tuple]:
	"""returns list of indices of pairs in `numbers` that their sum is `sum`"""
	complements_to_indices = {}  # value to sorted list of indices in numbers such that: numbers[idx] = value
	pairs_indices = []

	for idx, num in enumerate(numbers):
		if sum-num in complements_to_indices:
			for other_idx in complements_to_indices[sum-num]:
				pairs_indices.append((other_idx, idx))
		if num in complements_to_indices:
			complements_to_indices[num].append(idx)
		else:
			complements_to_indices[num] = [idx]
			
	return pairs_indices
	

@pytest.mark.parametrize("func", [pair_with_sum1, pair_with_sum2])
def test_pair_with_sum(random_ints: List[int], func):
	for sum in range(-10, 10):
		try:
			print("sum={}, result={}".format(sum, func(random_ints, sum)))
		except ValueError:
			pass


# Question 2


def subArrayZeroSum1(arr: List[int]) -> List[int]:
	"""returns first sub-array in array `arr` that it's sum is zero,
	if no such sub-array exists, returns None."""
	sum2index = {}
	sum = 0
	
	for idx, num in enumerate(arr):
		sum += num
		if sum in sum2index:
			return arr[sum2index[sum]+1:idx+1]
		else:
			sum2index[sum] = idx
			
	return None
	

# Question 3
	
	
def subArrayZeroSum2(arr: List[int]) -> List[Tuple]:
	"""returns list of all sub-arrays indices that their sum is zero"""
	sum2indices = {0: [0]}
	sum = 0
	subarrays = []
	
	for idx, num in enumerate(arr):
		sum += num
		if sum in sum2indices:
			for start_idx in sum2indices[sum]:
				subarrays.append((start_idx, idx+1))
		indices = sum2indices.get(sum, [])
		sum2indices[sum] = indices + [idx+1]
			
	return subarrays
	
	
def test_subArrayZeroSum() -> None:
	arr = [3, 4, -7, 3, 1, 3, 1, -4, -2, -2]
	print(subArrayZeroSum1(arr))
	print(subArrayZeroSum2(arr))
	
	
# Question 4


def sortedBinaryArray(arr: list) -> list:
	zero_counter = 0
	for num in arr:
		if num == 0:
			zero_counter += 1
	return [0 for _ in range(zero_counter)] + [1 for _ in range(len(arr)-zero_counter)]
	
	
def test_sortedBinaryArray() -> None:
	a = [0, 1, 0, 0, 1, 1, 1, 0]
	assert sortedBinaryArray(a) == [0, 0, 0, 0, 1, 1, 1, 1]
	
	
# Question 5


def findDuplicateIn(arr: List[int]) -> int:
	"""return value of duplicated item in `arr` including all integers from 0,1, ... to n-1"""
	arr_sum = sum(arr)
	n = len(arr)-1
	return arr_sum - (n * (n+1)) // 2


def test_findDuplicateIn() -> None:
	n = random.randint(0, 10)
	duplicate = random.randint(1, n)
	arr = [i+1 for i in range(n)] + [duplicate]
	random.shuffle(arr)
	print(arr)
	print(findDuplicateIn(arr))
	assert duplicate == findDuplicateIn(arr)


# Question 6


def consecutiveSubarrayIn(arr: List[int]) -> List[List[int]]:
	min_and_max = {(i, i): (arr[i], arr[i]) for i in range(len(arr))}
	solutions = []
	
	for s_idx in range(len(arr)):
		for e_idx in range(s_idx+1, len(arr)):
			# subarray from s_idx, s_idx+1, ... , e_idx (including e_idx)
			min_val, max_val = min_and_max[(s_idx, e_idx-1)]
			min_val = min(min_val, arr[e_idx])
			max_val = max(max_val, arr[e_idx])
			min_and_max[(s_idx, e_idx)] = min_val, max_val
			if e_idx-s_idx != max_val-min_val:
				continue
			subarray = set(arr[s_idx:e_idx+1])
			if len(subarray) == e_idx-s_idx+1:
				solutions.append(arr[s_idx:e_idx+1])
	
	return solutions


def test_consecutiveSubarrayIn() -> None:
	assert consecutiveSubarrayIn([2, 0, 2, 1, 4, 3, 1, 0]) == [[0, 2, 1], [0, 2, 1, 4, 3], [2, 1], [2, 1, 4, 3], [4, 3], [1, 0]]


# Question 7


def maxSubarrayWithSum(arr: List[int], sumOfInts: int) -> List[List[int]]:
	max_length, max_subarrays = 0, []
	
	for s_idx in range(len(arr)):
		for e_idx in range(s_idx+1, len(arr)+1):
			subarray = arr[s_idx:e_idx]
			if sum(subarray) == sumOfInts:
				if e_idx-s_idx > max_length:
					max_length = e_idx-s_idx
					max_subarrays = [subarray]
				elif e_idx-s_idx == max_length:
					max_subarrays.append(subarray)
	
	return max_subarrays


def test_maxSubarrayWithSum() -> None:
	assert maxSubarrayWithSum([], 0) == [] 
	assert maxSubarrayWithSum([1, 2, 3], 7) == []
	assert maxSubarrayWithSum([5, 6, -5, 5, 3, 5, 3, -2, 0], 8) == [[-5, 5, 3, 5]]
	assert maxSubarrayWithSum([-2, 0, 2, 0, -2], 0) == [[-2, 0, 2, 0], [0, 2, 0, -2]]


# Question 8


def sum0sAnd1s(arr: List[int]) -> int:
	return sum([1 if num == 1 else -1 for num in arr])


def maxSubarrayWithSame0sAnd1s1(arr: List[int]) -> List[List[int]]:
	max_length, max_subarrays = 0, []
	
	for s_idx in range(len(arr)):
		for e_idx in range(s_idx+1, len(arr)+1):
			subarray = arr[s_idx:e_idx]
			if sum0sAnd1s(subarray) == 0:
				if e_idx-s_idx > max_length:
					max_length = e_idx-s_idx
					max_subarrays = [subarray]
				elif e_idx-s_idx == max_length:
					max_subarrays.append(subarray)
	
	return max_subarrays


def maxSubarrayWithSame0sAnd1s2(arr: List[int]) -> List[List[int]]:
	temp_arr = list(-1 if i == 0 else 1 for i in arr)
	indices = subArrayZeroSum2(temp_arr)
	max_len = max(indices, key=lambda x: x[1]-x[0]+1)
	max_len = max_len[1]-max_len[0]+1
	indices = list(filter(lambda indices: indices[1]-indices[0]+1 == max_len, indices))
	return [arr[idx[0]:idx[1]] for idx in indices]


def test_maxSubarrayWithSame0sAnd1s() -> None:
	assert maxSubarrayWithSame0sAnd1s1([0, 0, 1, 0, 1, 0, 0]) == [[0, 1, 0, 1], [1, 0, 1, 0]]
	assert maxSubarrayWithSame0sAnd1s2([0, 0, 1, 0, 1, 0, 0]) == [[0, 1, 0, 1], [1, 0, 1, 0]]


# Question 9


def dutchSort(arr: List) -> None:
	counter = [0, 0, 0]
	for item in arr:
		counter[item] += 1
	for idx in range(len(arr)):
		if counter[0]:
			arr[idx] = 0
			counter[0] -= 1
		elif counter[1]:
			arr[idx] = 1
			counter[1] -= 1
		else:
			arr[idx] = 2


def test_dutchSort() -> None:
	arr = [0, 1, 2, 2, 1, 0, 0, 2, 0, 1, 1, 0]
	sorted_arr = sorted(arr)
	dutchSort(arr)
	assert arr == sorted_arr


# Question 10


def partialInsertionSort(arr: List, idx: int) -> None:
	for i in range(idx, len(arr)-1):
		if arr[i] > arr[i+1]:
			arr[i], arr[i+1] = arr[i+1], arr[i]
		else:
			break


def merge2arrays1(arr1: List, arr2: List) -> None:
	"""merge two sorted arrays in-place as arr1[i] < arr2[j]"""
	idx1 = 0

	while idx1 < len(arr1):
		if arr1[idx1] > arr2[0]:
			arr1[idx1], arr2[0] = arr2[0], arr1[idx1]
			partialInsertionSort(arr2, 0)
		idx1 += 1


@pytest.mark.parametrize(
	"a,b",
	[([1, 4, 6, 8], [2, 3, 5, 6, 7, 9]), ([1, 4, 7, 8, 10], [2, 3, 9])]
)
def test_merge2arrays1(a, b) -> None:
	sorted_ab = sorted(a+b)
	sorted_a = sorted_ab[:len(a)]
	sorted_b = sorted_ab[len(a):]
	merge2arrays1(a, b)
	assert a == sorted_a, a
	assert b == sorted_b, b


# Question 11


def merge2arrays2(arr1: List, arr2: List) -> None:
	"""merge two sorted arrays in-place at arr1. it is assumed:
		1. len(arr1) >= len(arr2).
		2. arr1 can contain all elements of arr1 and arr2 combined.
	"""
	lookup = 0
	while lookup < len(arr1) and arr1[lookup] is None:
			lookup += 1

	idx1, idx2 = 0, 0
	while lookup < len(arr1) and idx2 < len(arr2):
		# arr1 is sorted for all non-None values
		# arr1[:idx1] is sorted
		# arr1[lookup] is not None
		# arr2[idx2:] is sorted
		if arr1[lookup] > arr2[idx2]:
			arr1[idx1], arr2[idx2] = arr2[idx2], arr1[idx1]
			# arr1[:idx1+1] is sorted
			# arr2[idx2:] might not be sorted
			if arr2[idx2] is None:
				idx2 += 1
			else:
				partialInsertionSort(arr2, idx2)
			# arr2[idx2:] is sorted
		else: # arr1[lookup] <= arr2[idx2]
			arr1[idx1], arr1[lookup] = arr1[lookup], arr1[idx1]
			# arr1[:idx1+1] is sorted
			lookup += 1
		
		idx1 += 1
		# arr1[:idx1] is sorted
		while lookup < len(arr1) and (idx1 > lookup or arr1[lookup] is None):
			lookup += 1
	
	for idx in range(idx2, len(arr2)):
		arr1[idx1] = arr2[idx]
		idx1 += 1


@pytest.mark.parametrize(
	"a,b",
	[([None, 2, None, 3, None, 5, 6, None, None],
     [1, 8, 9, 10, 15]),
    ([1, None, 7, 8, None, None],
     [2, 3, 9])]
)
def test_merge2arrays2(a, b) -> None:
	sorted_a = sorted(filter(None, a+b))
	merge2arrays2(a, b)
	assert a == sorted_a, a


# Question 12


def max_consecutive_ones(array: List) -> int:
    """ procedural like style """
    if len(array) == 0: raise ValueError
    if len(array) == 1: return 0
    left_length = [0 for _ in range(len(array))]
    right_length = [0 for _ in range(len(array))]
    
    consecutive_ones = 0
    for idx in range(len(array)):
        if array[idx] == 1:
            consecutive_ones += 1
        else:
            consecutive_ones = 0
        left_length[idx] = consecutive_ones
    
    consecutive_ones = 0
    for idx in range(len(array) -1, -1, -1):
        if array[idx] == 1: 
            consecutive_ones += 1
        else:
            consecutive_ones = 0
        right_length[idx] = consecutive_ones

    max_idx, max_length = 0, right_length[1]
    if left_length[-2] > max_length:
        max_idx, max_length = len(array) - 1, left_length[-2]
    
    for idx in range(1, len(array) - 1):
        curr_length = left_length[idx - 1] + right_length[idx + 1]
        if curr_length > max_length:
            max_idx, max_length = idx, curr_length
    
    return max_idx


@pytest.mark.parametrize(
	"array,expected_index",
	[([0], 0), ([1], 0), (np.zeros(1, dtype=int), 0), (np.ones(1, dtype=int), 0),
     ([0, 0], 0), ([1, 1], 0), ([0, 1], 0), ([1, 0], 1),
     ([0, 1, 0, 0, 1, 1], 3), ([1, 1, 0, 0, 1, 0], 2),
     ([1, 1, 0, 1, 0, 0, 1, 1], 2),
    ]
)
def test_max_consecutive_ones(array: np.array, expected_index: int):
    actual_index = max_consecutive_ones(array)
    assert actual_index == expected_index


# Question 13


def shuffle_inplace(array: List[int]) -> List[int]:
    for idx in range(len(array) - 1):
        replace_idx = random.randint(idx, len(array) - 1)
        array[idx], array[replace_idx] = array[replace_idx], array[idx]
    return array


def test_shuffle_inplace(random_ints):
    print(random_ints)
    shuffled_ints = shuffle_inplace(random_ints)
    print(shuffled_ints)
