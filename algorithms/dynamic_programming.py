import typing
import pytest

from itertools import product, chain
from typing import List, Set


# Question 2


def lcs1(s1: str, s2: str) -> int:
	"""returns length of longest common sub-string of `s1` and `s2`"""
	n, m = len(s1), len(s2)
	if n == 0 or m == 0: return 0
	solutions = [[None for s2_idx in range(m)] for s1_idx in range(n)]
	
	# trivial solutions
	for s1_idx in range(n):
		solutions[s1_idx][0] = 1 if s1[s1_idx] == s2[0] else 0
	for s2_idx in range(m):
		solutions[0][s2_idx] = 1 if s1[0] == s2[s2_idx] else 0
		
	# iterate table and fill next solutions
	for s1_idx in range(1, n):
		for s2_idx in range(1, m):
			if s1[s1_idx] == s2[s2_idx]:
				solutions[s1_idx][s2_idx] = solutions[s1_idx-1][s2_idx-1] + 1
			else:
				solutions[s1_idx][s2_idx] = max(solutions[s1_idx][s2_idx-1], solutions[s1_idx-1][s2_idx])
				
	return solutions[n-1][m-1]


# Question 3


def lcs2(s1: str, s2: str) -> int:
	"""returns length of longest common sub-string of `s1` and `s2`
	improved space usage to one list instead of matrix"""
	n, m = len(s1), len(s2)
	if n == 0 or m == 0: return 0
	prev_solutions = [None for s1_idx in range(n)]
	
	for s1_idx in range(n):
		prev_solutions[s1_idx] = 1 if s1[s1_idx] == s2[0] else 0

	for s2_idx in range(1, m):
		for s1_idx in range(1, n):
			if s1[s1_idx] == s2[s2_idx]:
				prev_solutions[s1_idx] = prev_solutions[s1_idx-1] + 1
			else:
				prev_solutions[s1_idx] = max(prev_solutions[s1_idx-1], prev_solutions[s1_idx])
	
	return prev_solutions[n-1]


@pytest.mark.parametrize("lcs,s1,s2,expected_lcs",
	[tuple(chain((f, ), params)) for f, params in product([lcs1, lcs2],
		[("", "", 0),
		("", "abcd", 0),
		("a", "a", 1),
		("abc", "abc", 3),
		("ABCBDAB", "BDCABA", 4),
		]
	)]
)
def test_lcs(lcs, s1: str, s2: str, expected_lcs: int) -> None:
	assert expected_lcs == lcs(s1, s2)


# Question 4


def triple_lcs(s1: str, s2: str, s3: str) -> int:
	"""returns length of longest common sub-string of `s1`, `s2` and `s3`"""
	n, m, l = len(s1), len(s2), len(s3)
	if n == 0 or m == 0 or l == 0: return 0
	solutions = [[[None for s3_idx in range(l)] for s2_idx in range(m)] for s1_idx in range(n)]
	
	# trivial solutions
	for s1_idx in range(n):
		for s2_idx in range(m):
			for s3_idx in range(l):
				solutions[s1_idx][s2_idx][s3_idx] = 1 if s1[s1_idx] == s2[s2_idx] == s3[s3_idx] else 0
		
	# iterate table and fill next solutions
	for s1_idx in range(1, n):
		for s2_idx in range(1, m):
			for s3_idx in range(1, l):
				if s1[s1_idx] == s2[s2_idx] == s3[s3_idx]:
					solutions[s1_idx][s2_idx][s3_idx] = solutions[s1_idx-1][s2_idx-1][s1_idx-1] + 1
				else:
					solutions[s1_idx][s2_idx][s3_idx] = max(
						solutions[s1_idx-1][s2_idx][s3_idx],
						solutions[s1_idx][s2_idx-1][s3_idx],
						solutions[s1_idx][s2_idx][s3_idx-1],
						solutions[s1_idx-1][s2_idx-1][s3_idx],
						solutions[s1_idx][s2_idx-1][s3_idx-1],
						solutions[s1_idx-1][s2_idx][s3_idx-1])
				
	return solutions[n-1][m-1][l-1]


def test_triple_lcs() -> None:
	assert 4 == triple_lcs("ABCBDAB", "BDCABA", "BADACB")


# Question 5


def all_lcs_of(s1: str, s2: str) -> Set[str]:
	"""returns length of longest common sub-string of `s1` and `s2`
	improved space usage to one list instead of matrix"""
	n, m = len(s1), len(s2)
	if n == 0 or m == 0: return 0
	prev_solutions = [None for s1_idx in range(n)]
	next_solutions = [None for s1_idx in range(n)]
	
	for s1_idx in range(n):
		prev_solutions[s1_idx] = [s2[0]] if s1[s1_idx] == s2[0] else [""]
		next_solutions[s1_idx] = [s2[0]] if s1[s1_idx] == s2[0] else [""]

	for s2_idx in range(1, m):
		for s1_idx in range(1, n):
			if s1[s1_idx] == s2[s2_idx]:
				next_solutions[s1_idx] = [sol + s1[s1_idx] for sol in prev_solutions[s1_idx-1]]
			else:
				solutions = list(set(next_solutions[s1_idx-1] + prev_solutions[s1_idx]))
				max_len = len(max(solutions, key=len))
				next_solutions[s1_idx] = [solution for solution in solutions if len(solution) == max_len]
		prev_solutions = list(next_solutions)

	return set(next_solutions[n-1])


@pytest.mark.parametrize("s1,s2,expected_lcs",
	[("ABCBDAB", "BDCABA", {"BCBA", "BCAB", "BDAB"}),
	("XMJYAUZ", "MZJAWXU", {"MJAU"}),
	]
)
def test_all_lcs_of(s1: str, s2: str, expected_lcs: Set[str]) -> None:
	assert all_lcs_of(s1, s2) == expected_lcs