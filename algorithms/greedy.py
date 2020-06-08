import queue
import typing

from typing import List, Tuple
from collections import Counter, deque, namedtuple


# Question 1


def isOverlap(activity1: Tuple, activity2: Tuple) -> bool:
	return not (activity1[0] > activity2[1] or activity2[0] > activity1[1])


def maximizedActivities(activities: List[Tuple]) -> List[Tuple]:
	"""returns list of maximized number of activities that do not overlap"""
	if not activities:
		return []
	
	activities = sorted(activities, key=lambda activity: activity[1])  # sort by ending time
	
	included_activities = [activities[0]]
	idx = 1
	
	while idx < len(activities):
		if not isOverlap(activities[idx], included_activities[-1]):
			included_activities.append(activities[idx])
		idx += 1
		
	return included_activities
	
	
def test_maximizedActivities() -> None:
	arr = [(1, 4), (3, 5), (0, 6), (5, 7), (3, 8), (5, 9), (6, 10), (8, 11), (8, 12), (2, 13), (12, 14)]
	assert maximizedActivities(arr) == [(1, 4), (5, 7), (8, 11), (12, 14)]
	
	
# Question 2


class HuffmanNode:
	
	def __init__(self, sum: int, left=None, right=None, letter=None):
		self.sum = sum
		self.letter = letter
		self.left = left
		self.right = right
		
	def __str__(self):
		return "sum={}, letter={}".format(self.sum, self.letter)
	
	def __gt__(self, other):
		return self.sum > other.sum
	
	def __eq__(self, other):
		return self.sum == other.sum


def encodeHuffman(s: str) -> str:
	s = ''.join(c.lower() for c in s)
	c = Counter(s)
	nodes = [HuffmanNode(sum=count, letter=letter) for letter, count in c.items()]
	q = queue.PriorityQueue()
	for item in nodes:
		q.put(item)
	while q.qsize() > 1:
		current_node1 = q.get()
		current_node2 = q.get()
		new_node = HuffmanNode(current_node1.sum+current_node2.sum, current_node2, current_node1)
		q.put(new_node)
	letter2code = {}
	q = deque([(q.get(), "")])
	while q:
		current_node, current_code = q.popleft()
		if current_node.left is None and current_node.right is None:
			letter2code[current_node.letter] = current_code
		else:
			if current_node.left is not None:
				q.append((current_node.left, current_code+"0"))
			if current_node.right is not None:
				q.append((current_node.right, current_code+"1"))
	return ''.join(letter2code[letter] for letter in s)
	

def test_encodeHuffman() -> None:
	assert encodeHuffman("aaabbc") == "111000001"


# Question 3


def isOverlapBetween(s1: str, s2: str, overlap: int) -> bool:
	for idx in range(overlap):
		if s1[idx] != s2[len(s2)-overlap+idx]:
			return False
	return True


def maxOverlapBetween(s1: str, s2: str) -> int:
	len1, len2 = len(s1), len(s2)
	max_overlap = 0
	overlap = 1
	while overlap <= min(len1, len2):
		if isOverlapBetween(s1, s2, overlap):
			max_overlap = overlap
		overlap += 1
	return max_overlap


def minimizeSuperstringOf(strings: List[str]) -> str:
	
	while len(strings) > 1:
		max_overlap, max_pair = -1, (-1, -1)
		for idx in range(len(strings)-1):
			overlap = maxOverlapBetween(strings[idx], strings[idx+1])
			if overlap > max_overlap:
				max_overlap, max_pair = overlap, (idx, idx+1)
			overlap = maxOverlapBetween(strings[idx+1], strings[idx])
			if overlap > max_overlap:
				max_overlap, max_pair = overlap, (idx+1, idx)
		pair = strings[max_pair[0]], strings[max_pair[1]]
		merged_string = pair[1][:] + pair[0][max_overlap:]
		strings.pop(min(max_pair))
		strings[min(max_pair)] = merged_string
		
	return strings[0]


def test_minimizeSuperstringOf() -> None:
	strings = ["CATGC", "CTAAGT", "GCTA", "TTCA", "ATGCATC"]
	superstring = minimizeSuperstringOf(list(strings))
	for s in strings:
		assert s in superstring
	assert (sum(len(s) for s in strings)) >= len(superstring)


# Question 4


Task = namedtuple('Task', ['deadline', 'profit'])


def maxProfitFrom(tasks: List[Task]) -> int:
	slots_free = [True for _ in range(max(tasks, key=lambda tasks: tasks.deadline).deadline)]
	tasks = sorted(tasks, key=lambda task: task.profit)
	total_profit = 0
	while tasks:
		deadline, profit = tasks.pop()
		next_slot = None
		for idx in range(deadline-1, -1, -1):
			if slots_free[idx]:
				next_slot = idx
				break
		if next_slot is not None:
			total_profit += profit
			slots_free[next_slot] = False
	return total_profit


def test_maxProfitFrom() -> None:
	assert maxProfitFrom([Task(3, 5), Task(2, 6), Task(9, 7), Task(3, 0), Task(4, 4)]) == 22
	assert maxProfitFrom([Task(9, 15), Task(2, 2), Task(5, 18), Task(7, 1), Task(4, 25), Task(2, 20), Task(5, 8), Task(7, 10), Task(4, 12), Task(3, 5)]) == 109


# Question 5


def minimumColorsIn(graph: dict) -> int:
	root = list(graph.keys())[0]
	vertices2colors = {}
	stack = deque([root])
	seen = set([root])
	colors_count = 0
	
	while stack:
		current_vertice = stack.popleft()
		available_colors = set([color_idx for color_idx in range(colors_count)])
		for neighbour_vertice in graph[current_vertice]:
			if neighbour_vertice in vertices2colors:
				available_colors -= set([vertices2colors[neighbour_vertice]])
			if neighbour_vertice not in seen:
				seen.add(neighbour_vertice)
				stack.append(neighbour_vertice)
		if not available_colors:
			vertices2colors[current_vertice] = colors_count
			colors_count += 1
		else:
			vertices2colors[current_vertice] = min(available_colors)
	
	return colors_count


def test_minimumColorsIn() -> None:
	graph = {"A": ["B", "E", "F"], "B": ["A", "D", "E"], "C": ["D", "E"], "D": ["B", "C"], "E": ["A", "B", "C", "F"], "F": ["A", "E"]}
	assert minimumColorsIn(graph) == 3


# Question 6


def kruskal(vertices: List[int], edges: List[Tuple]) -> int:
	edges_queue = queue.PriorityQueue()
	for edge in edges:
		edges_queue.put(edge)
	vertice2group = [vertice for vertice in range(len(vertices))]
	group_count = [1 for vertice in range(len(vertices))]
	total_weight = 0
	
	while edges_queue.qsize():
		weight, vertice1, vertice2 = edges_queue.get()
		if vertice2group[vertice1] != vertice2group[vertice2]:
			total_weight += weight
			# on different groups - need to be merged_string
			if group_count[vertice2group[vertice1]] < group_count[vertice2group[vertice2]]:
				# make sure that `vertice1`'s group is greater equal to that of `vertice2`
				vertice1, vertice2 = vertice2, vertice1
			group_count[vertice2group[vertice2]] = 0
			# update groups
			for vertice in vertices:
				if vertice2group[vertice] == vertice2group[vertice2]:
					vertice2group[vertice] = vertice2group[vertice1]
					group_count[vertice2group[vertice1]] += 1
	
	return total_weight


def test_kruskal() -> None:
	assert kruskal(
		[0, 1, 2, 3, 4, 5, 6],
		[(7, 0, 1), (5, 0, 3), (8, 1, 2), (7, 1, 4), (9, 1, 3), (15, 2, 4), (6, 3, 5), (5, 2, 4), (8, 4, 5), (9, 4, 6), (11, 5, 6)]) == 39


test_kruskal()