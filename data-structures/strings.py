import random
import pytest
from collections import defaultdict
from typing import List


# Utilites


def random_string(min_length: int = 0, max_length: int = 10) -> str:
	letters = [chr(value) for value in range(ord('a'), ord('z')+1)]
	return "".join(random.choice(letters) for i in range(random.randint(min_length, max_length)))


# Question 1
## Version 1


def isPalindrom1(s: str) -> bool:
	low, high = 0, len(s)-1
	while low < high:
		if s[low] != s[high]:
			return False
		low += 1
		high -= 1
	return True
	
	
def isRotatedPalindrom1(s: str) -> bool:
	for start_idx in range(len(s)):
		if isPalindrom1(s[start_idx:] + s[:start_idx]):
			return True
	return False
	
	
## Version 2


def isPalindrom2(s: str, low: int, high: int) -> bool:
	while low < high:
		if s[low] != s[high]:
			return False
		low += 1
		high -= 1
	return True


def isRotatedPalindrom2(s: str) -> bool:
	for low in range(len(s)):
		if isPalindrom2(s+s, low, low+len(s)-1):
			return True
	return False
	
	
## Version 3
	

def isRotatedPalindrom3(s: str) -> bool:
	if not s: return True
	for left in range(len(s)):
		is_palindrom = True
		right = (left-1) % len(s)
		for offset in range(len(s)//2 + 1):
			if s[(left+offset) % len(s)] != s[(right-offset) % len(s)]:
				is_palindrom = False
				break
		if is_palindrom:
			return True
	return False
	
	
def test_isRotatingPalindrom() -> None:
	strings = [
		"CBAABCD",
		"BAABCC",
		"AAA",
		"AA",
		"",
		"ABC",
	]
	for s in strings:
		print("{}: {}".format(s, isRotatedPalindrom2(s)))
		
		
# Question 2


def longestPalindromAt(s: str, idx: int, odd: int) -> str:
	"""returns longest palindrom in str `s` starting at index `idx`"""
	offset = 0
	while idx - offset >= 0 and idx + offset + odd < len(s):
		if s[idx-offset] != s[idx+offset+odd]:
			offset -= 1
			break
		offset += 1
	return s[idx-offset:idx+offset+odd+1]
	
	
def longestPalindrom(s: str) -> str:
	longest_pal = ""
	start_idx = 0
	
	while start_idx < len(s):
		pal1, pal2 = longestPalindromAt(s, start_idx, 0), longestPalindromAt(s, start_idx, 1)
		pal = pal1 if len(pal1) > len(pal2) else pal2
		if len(pal) > len(longest_pal):
			longest_pal = pal
		start_idx += 1
		
	return longest_pal
	
	
def test_longestPalindrom() -> None:
	strings = ["bananas", "abdcbcdbdcbbc"]
	for s in strings:
		print(longestPalindrom(s))
		
		
# Question 3
## Version 1 - incorrect (returns re-occuring substrings) and not finished


def pairs2indicesFrom(s: str) -> {}:
	"""returns dictionary of two characters in `s` to their starting indices"""
	if not s:
		return {}
	
	pairs2indices = {}
	prev = s[0]
	
	for idx, c in enumerate(s[1:]):
		indices = pairs2indices.get((prev, c), [])
		indices.append(idx)
		pairs2indices[(prev, c)] = indices
		prev = c
		
	return pairs2indices
	
	
def test_pairs2indices() -> None:
	strings = [random_string() for _ in range(10)]
	
	for s in strings:
		print("pairs2indicesFro({})={}".format(s, pairs2indicesFrom(s)))
		
		
def partitionListByFunc(l: List, func) -> List[List]:
	values = set([func(item) for item in l])
	return {value: [item for item in l if func(item) == value] for value in values}
		
		
def modulo5func(value) -> int:
	return ord(value) % 5
		
		
def test_partitionListByFunc() -> None:
	strings = [random_string() for _ in range(10)]
	
	for s in strings:
		print(s)
		partition = partitionListByFunc(s, modulo5func)
		print(partition)
		
		
def longestSequenceAt(s: str, indices: List[int], offset: int = 0) -> List[str]:
	ref_idx = indices[0]
	max_idx = max(indices)
	
	while max_idx+offset < len(s) and all(s[ref_idx+offset] == s[idx+offset] for idx in indices):
		offset += 1
		
	sequences = [s[ref_idx:ref_idx+offset]]
	# partition indices according to next characters
	next_chars = set(s[idx+offset] for idx in indices if idx+offset < len(s))
	partition = {next_char: [idx for idx in indices if idx+offset < len(s) and s[idx+offset] == next_char] for next_char in next_chars}
	partition = {next_char: indices for next_char, indices in partition.items() if len(indices) > 1}
	
	for indices_subset in partition.values():
		sequences.extend(longestSequenceAt(s, indices_subset, offset))
		
	return sequences
	
	
def allSubSequencesOf(s: str) -> List[str]:
	pairs2indices = pairs2indicesFrom(s)
	subsequences = []
	
	for indices in pairs2indices.values():
		if len(indices) > 1:
			subsequences.extend(longestSequenceAt(s, indices, 2))
		
	return subsequences
	
	
def test_allSubSequencesOf() -> None:
	strings = [random_string(5, 15) for _ in range(10)]
	
	for s in strings:
		print(s)
		print(allSubSequencesOf(s))


## Version 2 - checks whether they exist


def isRepeatingSubSequencesIn(s: str) -> bool:
	char_fruquencies = {}
	
	for c in s:
		freq = char_fruquencies.get(c, 0)
		freq += 1
		if freq == 3:
			return True
		char_fruquencies[c] = freq
		
	new_s = ''.join(str(c) for c in s if char_fruquencies[c] > 1)
	
	return not isPalindrom2(new_s, 0, len(new_s)-1)
	
	
def test_isRepeatingSubSequencesIn() -> None:
	input_and_expected = [
		("XYBAXB", True),
		("XBXAXB", True),
		("ABCA", False),
		("XYBYAXBY", True),
		]
	
	for input, expected in input_and_expected:
		assert isRepeatingSubSequencesIn(input) == expected
		
		
# Question 4


def isRotationOfAt(s1: str, s2: str, s_idx: int) -> bool:
	for diff in range(len(s1)):
		curr_idx = (s_idx + diff) % len(s1)
		if s2[diff] != s1[curr_idx]:
			return False
	return True


def isRotationOf(s1: str, s2: str) -> bool:
	if len(s1) != len(s2): return False
	if not s1: return True
	
	for s_idx in range(len(s1)):
		if isRotationOfAt(s1, s2, s_idx):
			return True
	
	return False


def test_isRotationOf() -> None:
	assert isRotationOf("", "")
	assert not isRotationOf("", "a")
	assert not isRotationOf("bb", "a")
	assert not isRotationOf("b", "a")
	assert isRotationOf("a", "a")
	assert isRotationOf("abc", "cab")


# Question 5


def isCircural(moves: str) -> bool:
	if not moves: return True
	if any(move not in ['M', 'E', 'N', 'W', 'S'] for move in moves):
		raise ValueError("all moves must be one of the types: M, E, N, W, S")
	if moves[0] == 'M': raise ValueError("moves must start with a direction.")
	x_y = (0, 0)
	for move in moves:
		if move == 'E':
			current_diff = (1, 0)
		elif move == 'N':
			current_diff = (0, 1)
		elif move == 'W':
			current_diff = (-1, 0)
		elif move == 'S':
			current_diff = (0, -1)
		else:
			x_y = (x_y[0] + current_diff[0], x_y[1] + current_diff[1])
	return x_y == (0, 0)


def test_isCircural() -> None:
	assert isCircural("")
	assert isCircural("NMSM")
	assert not isCircural("NM")
	assert not isCircural("NMMSM")
	assert isCircural("WMSEM")


# Question 6


def num2excelColumn(num: int) -> str:
	if num <= 0: raise ValueError
	
	BASE = ord('Z')-ord('A')+1
	OFFSET = ord('A')
	
	letters = []
	while num > 0:
		div, mod = num // BASE, (num-1) % BASE
		
		letters.append(chr(mod + OFFSET))
		num = (num - 1) // BASE
		if div == 0:
			break
	
	return ''.join(letters[::-1])


def num2excelColumn_assert(num: int, column: str) -> None:
	assert num2excelColumn(num) == column, "expected {}->{}, got {}".format(num, column, num2excelColumn(num))


def test_num2excelColumn() -> None:
	num2excelColumn_assert(1, "A")
	num2excelColumn_assert(7, "G")
	with pytest.raises(ValueError):
		num2excelColumn(0)
		num2excelColumn(-100)
	num2excelColumn_assert(26, "Z")
	num2excelColumn_assert(27, "AA")
	num2excelColumn_assert(105, "DA")
	num2excelColumn_assert(701, "ZY")
	num2excelColumn_assert(1015, "AMA")


# Question 7


def are_anagrams0(text1: str, text2: str) -> bool:
  if not all(isinstance(text, str) for text in [text1, text2]):
    raise ValueError(f"Both text1 and text2 must be strings")

  chars_count1, chars_count2 = defaultdict(int), defaultdict(int)

  for c1 in text1:
    if c1.isalpha():
      chars_count1[c1.lower()] += 1

  for c2 in text2:
    if c2.isalpha():
      chars_count2[c2.lower()] += 1

  return chars_count1 == chars_count2


def _count_alphas(text: str) -> defaultdict:
  chars_count = defaultdict(int)

  for c in text:
   if c.isalpha():
     chars_count[c.lower()] += 1

  return chars_count


def are_anagrams1(text1: str, text2: str) -> bool:
  if not all(isinstance(text, str) for text in [text1, text2]):
    raise ValueError(f"Both text1 and text2 must be strings")

  chars_count1 = _count_alphas(text1)
  chars_count2 = _count_alphas(text2)

  return chars_count1 == chars_count2


def are_anagrams2(text1: str, text2: str) -> bool:
  if not all(isinstance(text, str) for text in [text1, text2]):
    raise ValueError(f"Both text1 and text2 must be strings")

  chars_count = defaultdict(int)

  for c1 in text1:
    if c1.isalpha():
      chars_count[c1.lower()] += 1

  for c2 in text2:
    if c2.isalpha():
      chars_count[c2.lower()] -= 1
      if chars_count[c2.lower()] < 0:
        return False

  return len(chars_count) == 0


def are_anagrams3(text1: str, text2: str) -> bool:
  if not all(isinstance(text, str) for text in [text1, text2]):
    raise ValueError(f"Both text1 and text2 must be strings")

  alphas1 = set(filter(str.isalpha, text1))
  alphas2 = set(filter(str.isalpha, text1))

  return alphas1 == alphas2


def inputs(*func_inputs):
  def func_decorator(func):
    def inner():
      for func_args in func_inputs:
        try:
          func(*func_args)
        except Exception as e:
          print(e)
    return inner
  return func_decorator


@inputs(
  ('', 'a', False),
  ('a', 'b', False),
  ('a', 'a', True),
  ('ab', 'ab', True),
  ('ab', 'ba', True),
  ('aab', 'aba', True),
  ('abb', 'aab', False),
  ('I\'m Lord Voldemort', 'Tom Riddle mlorvo', True),
  ('24234', '', True),
)
def test_valid_inputs(
    text1: str, text2: str, expected_are_anagrms: bool,
):
  assert are_anagrams3(text1, text2) == expected_are_anagrms

@inputs(
  (1, 'text'),
  (123, '4343'),
  (str, zip),
)
def test_invalid_inputs(text1, text2):
  # this is a method in part of TestCase class
  # we could make a custom ValueError just to be more accurate
  with self.assertRaises(ValueError):
    are_anagrams(text1, text2)

