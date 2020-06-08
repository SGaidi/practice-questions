import typing

from typing import List, Tuple


# Question 1


def isValidPartialSolution(col_indices: List[int], new_indices: Tuple) -> bool:
	
	y, x = new_indices
	
	for other_y, other_x in enumerate(col_indices):

		# Same column or row
		if other_x == x or other_y == y:
			return False
			
		# Same diagnal
		if abs(x - other_x) == y - other_y:
			return False
			
	return True


def printQueensSolutions(queen_count: int = 8, row: int = 0, col_indices: List[int] = None) -> None:
	
	if col_indices is None:
		col_indices = []
	
	if queen_count < 4:
		raise ValueError("No solutions exist for {} queens problem".format(queen_count))

	if row == queen_count:
		print('  ' + " ".join(str(col) for col in range(queen_count)))
		for row, col in enumerate(col_indices):
			print(str(row) + ':' + "  " * col + '*')
			
	for col in range(queen_count):
		if isValidPartialSolution(col_indices, (row, col)):
			col_indices.append(col)
			printQueensSolutions(queen_count, row+1, col_indices)
			col_indices.pop()
			
			
def test_printQueensSolutions() -> None:
	for queen_count in range(4, 9):
		printQueensSolutions(queen_count)


# Question 2


def is_valid_indices(x: int, y: int, rows: int, cols: int) -> bool:
	return x >= 0 and x < rows and y >= 0 and y < cols


def valid_indices_from(x: int, y: int, curr_path: list, rows: int, cols: int) -> List[Tuple[int, int]]:
	return [(new_x, new_y) for (new_x, new_y) in
		[(x+1, y+2), (x+1, y-2), (x+2, y+1), (x+2, y-1), (x-1, y+2), (x-1, y-2), (x-2, y+1), (x-2, y-1)]
		if is_valid_indices(new_x, new_y, rows, cols) and (new_x, new_y) not in curr_path]


def knight_travels_from(x: int, y: int, rows: int, cols: int) -> List[List[Tuple[int, int]]]:
	END_OF_LAYER = None

	cells = [(x, y)]
	curr_path = []
	paths = []
	
	while cells:
		
		next_cell = cells.pop()
		if next_cell == END_OF_LAYER:
			curr_path.pop()
			continue

		curr_x, curr_y = next_cell
		curr_path.append((curr_x, curr_y))
		next_indices = valid_indices_from(curr_x, curr_y, curr_path, rows, cols)
		
		if not next_indices:
			# path is complete
			if len(curr_path) == rows * cols:
				# path covers all matrix
				paths.append(list(curr_path))
				
			# remove last
			assert (curr_x, curr_y) == curr_path.pop(), "Expected: {}, {}".format(curr_x, curr_y)
		else:
			# path continues
			cells.append(END_OF_LAYER)
			for next_x, next_y in next_indices:
				cells.append((next_x, next_y))
	
	return paths


def knight_travels(rows: int = 8, cols: int = 8) -> List[List[Tuple[int, int]]]:
	paths = []
	
	for start_x in range(rows):
		for start_y in range(cols):
			paths.extend(knight_travels_from(start_x, start_y, rows, cols))
	
	return paths


def test_knight_travels() -> None:
	assert [] == knight_travels(4, 4)
	knight5tours = knight_travels(5, 5)
	assert len(knight5tours) == 1728, "Invalid tours count: {}".format(len(knight5tours))


test_knight_travels()