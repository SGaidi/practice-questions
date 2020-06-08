import random
import typing
import pytest

from typing import List, Tuple


# Utilites


def random_ints() -> List[int]:
	return [random.randint(-20, 20) for count in range(10)]


# Question 1


def insertion_sort(a: list) -> None:
	for right_idx in range(1, len(a)):
		current_idx = right_idx
		while current_idx > 0 and a[current_idx] < a[current_idx-1]:
			a[current_idx], a[current_idx-1] = a[current_idx-1], a[current_idx]
			current_idx -= 1


# Question 2


def selection_sort(a: list) -> None:
	for left_idx in range(-1, len(a)):
		min_idx, min_val = len(a)-1, a[-1]
		for right_idx in range(len(a)-2, left_idx, -1):
			if a[right_idx] < min_val:
				min_idx, min_val = right_idx, a[right_idx]
		for right_idx in range(min_idx, left_idx+1, -1):
			a[right_idx], a[right_idx-1] = a[right_idx-1], a[right_idx]


# Question 3


def bubble_sort(a: list) -> None:
	for e_idx in range(len(a)-1, 0, -1):
		for s_idx in range(e_idx):
			if a[s_idx] > a[s_idx+1]:
				a[s_idx], a[s_idx+1] = a[s_idx+1], a[s_idx]


# Question 4


def merge_sort(a: list, low: int = 0, high: int = None) -> None:
	if high is None: high = len(a)
	
	if low + 2 == high:
		if a[low] > a[high-1]:
			a[low], a[high-1] = a[high-1], a[low]
	elif low + 2 < high:
		mid = (low + high) // 2
		merge_sort(a, low, mid)
		merge_sort(a, mid, high)
		
		l_tmp, r_tmp = low, mid
		merged_arr = []
		while l_tmp < mid and r_tmp < high:
			if a[r_tmp] < a[l_tmp]:
				merged_arr.append(a[r_tmp])
				r_tmp += 1
			else:
				merged_arr.append(a[l_tmp])
				l_tmp += 1
		merged_arr.extend(a[r_tmp:high])
		merged_arr.extend(a[l_tmp:mid])
		a[low:high] = merged_arr


@pytest.mark.parametrize("func", [insertion_sort, selection_sort, bubble_sort, merge_sort])
def test_sort(func) -> None:
	a = random_ints()
	b = list(a)
	a.sort()
	func(b)
	assert a == b
