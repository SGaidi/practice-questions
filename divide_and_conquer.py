import random
import typing
import pytest

from typing import List


# Utilites


def random_ints() -> List[int]:
	return [random.randint(-10, 10) for count in range(20)]
	
	
# Question 1


def binarySearch(arr: list, value, low: int = 0, high: int = None) -> int:
	"""returns index of `value` in `arr`"""
	if high is None: high = len(arr)-1
	
	while low <= high:
		mid = (low+high)//2
		if arr[mid] == value:
			return mid
		elif arr[mid] > value:
			high = mid-1
		else:  # arr[mid] < value
			low = mid+1
	
	raise ValueError("value {} does not exist in arr".format(value))
	

@pytest.fixture
def sorted_arr() -> List[int]:
	return sorted(random_ints())


@pytest.fixture
def circular_sorted_arr(sorted_arr) -> List[int]:
	circulation = rand.randint(0, len(sorted_arr)-1)
	return sorted_arr[circulation:] + sorted_arr[:circulation]


@pytest.fixture
def value() -> int:
	return random.randint(-20, 20)


def test_binarySearch(sorted_arr: List[int], value: int) -> None:
	if value in sorted_arr:
		indices = []
		start = 0
		while sorted_arr.index(value, start):
			indices.append(sorted_arr.index(value, start))
			start = sorted_arr.index(value, start) + 1
		assert binarySearch(sorted_arr, value) in indices
	else:
		with pytest.raises(ValueError):
			binarySearch(sorted_arr, value)
		
		
# Question 2


def rotationsCountOf(arr: List[int]) -> int:
	low, high = 0, len(arr)-1
	if arr[low] <= arr[high]: return 0
	
	while low+1 < high:
		mid = (low+high) // 2
		if arr[low] <= arr[mid]:
			low = mid
		else:
			high = mid
	
	return low+1


@pytest.mark.parametrize("arr,expected_rotations", [
	([8, 9, 10, 2, 5, 6], 3),
	([-2, -1, 0, 2, 5, 6], 0),
	([1, 2, 0], 2),
	])
def test_rotationsCountOf(arr: List[int], expected_rotations: int) -> None:
	assert rotationsCountOf(arr) == expected_rotations


# Question 3


def searchInRotated(arr: List[int], value: int) -> int:
	rotations = rotationsCountOf(arr)
	if rotations:
		try:
			return binarySearch(arr, value, 0, rotations-1)
		except ValueError:
			pass
		return binarySearch(arr, value, rotations)
	else:
		return binarySearch(arr, value)


@pytest.mark.parametrize("circular_sorted_arr", [
	[8, 9, 10, 2, 5, 6],
	[-2, -1, 0, 2, 5, 6],
	[1, 2, 0],
	])
def test_searchInRotated(circular_sorted_arr: List[int]) -> None:
	for idx in range(len(circular_sorted_arr)):
		assert searchInRotated(circular_sorted_arr, circular_sorted_arr[idx]) == idx