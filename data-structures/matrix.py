import random
import typing
import pytest
import numpy as np

from typing import List


# Utilities


def random_numbers(count: int = 10) -> List[int]:
	return [random.randint(-10, 10) for _ in range(count)]
	
	
def random_square_matrix(count: int = 10) -> np.ndarray:
	return np.array([random_numbers(count) for _ in range(count)])
	
	
# Question 1


def spiral_of(matrix: np.ndarray) -> np.ndarray:
	return np.array([row if idx % 2 == 0 else row[::-1] for idx, row in enumerate(matrix)])


@pytest.mark.parametrize("array,expected_spiral", [
	(np.array([]), np.array([])),
	(np.array([[1]]), np.array([[1]])),
	(np.array([[1, 2]]), np.array([[1, 2]])),
	(np.array([[1, 2], [3, 4]]), np.array([[1, 2], [4, 3]])),
	(np.array([[1, 2, 3], [4, 5, 6]]), np.array([[1, 2, 3], [6, 5, 4]])),
	(np.array([[1, 2], [3, 4], [5, 6]]), np.array([[1, 2], [4, 3], [5, 6]])),
	(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]), np.array([[1, 2, 3], [6, 5, 4], [7, 8, 9], [12, 11, 10]])),
	])
def test_spiral_of(array: np.array, expected_spiral: np.array) -> None:
	assert np.array_equal(expected_spiral, spiral_of(array)), "input: {}, expected: {}, actual: {}".format(array, expected_spiral, spiral_of(array))
	
	
# Question 2


def array2spiral_matrix(arr: List[int], n: int, m: int) -> List[List[int]]:
	assert n * m == len(arr)
	matrix = []
	for row_idx in range(n):
		if row_idx % 2 == 0:
			matrix.append(arr[row_idx * m:(row_idx + 1) * m ])
		else:
			matrix.append(arr[(row_idx + 1) * m - 1:row_idx * m - 1:-1])
	return matrix


def test_array2spiral_matrix() -> None:
	n, m = random.randint(0, 10), random.randint(0, 10)
	print("{},{}".format(n, m))
	arr = [i+1 for i in range(n*m)]
	for row in array2spiral_matrix(arr, n, m):
		print(row)

# Question 3


def spiralShift(matrix: List[List[int]]) -> None:
	temp = matrix[0][0]
	for row in range(len(matrix)):
		for col in range(len(matrix[0])):
			if row == 0 and col == 0:
				temp = matrix[0][0]
				matrix[0][0] = matrix[-1][-1]
			else:
				temp, matrix[row][col] = matrix[row][col], temp


def test_spiralShift() -> None:
	n, m = random.randint(1, 10), random.randint(1, 10)
	matrix = [[i + 1 + (j * m) for i in range(m)] for j in range(n)]
	print(matrix)
	spiralShift(matrix)
	for row in matrix:
		print(row)


# Question 4


def random_matrix(size: int) -> np.array:
    return np.random.randint(0, size, size=(size, size))


def mat_djikstra():
    map_size = 10
    my_map = random_matrix(map_size)
    source_indices = (0, 0)
    target_indices = (map_size-1, map_size-1)
    
    indices_to_length = {}
    unseen_indices = [source_indices]  # queue
    
    while unseen_indices:
        current_indices = unseen_indices.pop(0)
        row, col = current_indices
        steps = my_map[row][col]
        next_indices = (
            (row + steps, col),
            (row, col + steps),
            (row - steps, col),
            (row, col - steps),
        )
        
        for next_row, next_col in next_indices:
            if next_row < 0 or next_row >= len(my_map):
                continue
            if next_col < 0 or next_col >= len(my_map[0]):
                continue
            
            current_length = indices_to_length.get(current_indices, 0)
            best_next_length = indices_to_length.get((next_row, next_col), float('inf'))
            
            if best_next_length == float('inf') or best_next_length > current_length + steps:
                indices_to_length[(next_row, next_col)] = current_length + steps
                unseen_indices.append((next_row, next_col))
    
    if target_indices not in indices_to_length:
        raise ValueError("not found")
    
    return indices_to_length[target_indices]


