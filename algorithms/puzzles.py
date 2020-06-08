import typing
import random
import pytest

from typing import List, Set


# Question 1


def angleOf(hour: int, minute: int) -> int:
	"""returns degrees between hour and minute hands in analog clock"""
	angle = abs(30 * hour - 5.5 * minute)
	if angle > 180:
		angle = 360 - angle
	return angle
	

@pytest.mark.parametrize("hour,minute,expected_angle",
	[(5, 30, 15.0), (9, 0, 90.0), (12, 0, 0.0)])
def test_angleOf(hour: int, minute: int, expected_angle: float) -> None:
	assert angleOf(hour, minute) == expected_angle
		
		
# Question 2


def test_addition() -> None:
	a, b = random.sample([1,10], 2)
	assert a + b == a -(-b)


# Question 3


def power_set_of(arr: List[int]) -> List[Set[int]]:
	sets = [set([])]
	for item in arr:
		new_sets = []
		for old_set in sets:
			new_sets.append(old_set)
			new_sets.append(old_set | {item})
		sets = new_sets
	return sets


@pytest.mark.parametrize("s,power_s", [
	([], [set([])]),
	([1], [set([]), {1}]),
	([1, 2], [set([]), {1}, {2}, {1, 2}]),
	])
def test_power_set_of(s: List[int], power_s: List[List[int]]):
	generated_power_set = power_set_of(s)
	for new_set in generated_power_set:
		assert new_set in power_s
	for expected_set in power_s:
		assert expected_set in generated_power_set