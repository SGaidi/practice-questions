import pytest
import itertools
from typing import List, Tuple, Set, Hashable, Generic, Any, TypeVar, Iterable
from abc import ABCMeta, abstractmethod


class Comparable(metaclass=ABCMeta):

    @abstractmethod
    def __lt__(self, other: Any) -> bool: ...


# Questions 1  2 - Terminology and Representations of Graphs

## Version 1 - Minimal Presentation


class MinimalGraph:
    """Directed graph, allows loops, but no duplicate nodes or edges.

    The node and edge definition is dependant of the objects equality operator
    implementation and hash value.

    Most applications won't use these types of graphs, as there would be
    additional constraints and relationships between the nodes and edges
    values, depending on the problems it seeks to solve."""

    nodes: set
    edges: set

    def __init__(self):
        self.nodes = set([])
        self.edges = set([])

    def __str__(self):
        return f'MinimalGraph(nodes={self.nodes}; edges={self.edges})'

    def add_node(self, node) -> None:
        if node in self.nodes:
            raise ValueError(f'Node {node} already exists.')
        self.nodes.add(node)

    def add_edge(self, node_from, node_to) -> None:
        edge = (node_from, node_to)
        if (node_from not in self.nodes) or (node_to not in self.nodes):
            raise ValueError(f'Edge {edge} must be between nodes in graph.')
        if edge in self.edges:
            raise ValueError(f'Edge {edge} already exists.')
        self.edges.add(edge)


class TestMinimalGraph:

    @pytest.mark.parametrize(
        "nodes_list",
        [[(1, 2, 3, 4), ],
         [(1,), (2, 3), (4, 5, 6)],
         [(1, 2.0, "3", object, MinimalGraph, set), ]]
    )
    def test_add_node(self, nodes_list: List[Tuple]):
        graph = MinimalGraph()
        expected_nodes = []

        for nodes in nodes_list:
            for node in nodes:
                graph.add_node(node)
                expected_nodes.append(node)
            assert set(expected_nodes) == graph.nodes

    def test_add_node_that_already_exists(self):
        graph = MinimalGraph()
        graph.add_node(1)
        with pytest.raises(ValueError):
            graph.add_node(1)

    def test_add_edge(self):
        graph = MinimalGraph()
        for node in range(1, 4):
            graph.add_node(node)
        for edge in itertools.product(range(1, 4), repeat=2):
            graph.add_edge(*edge)

    def test_add_edge_not_in_nodes(self):
        graph = MinimalGraph()
        graph.add_node(1)
        graph.add_node(2)

        for edge in ((1, 3), (3, 1), (3, 4)):
            with pytest.raises(ValueError):
                graph.add_edge(*edge)

    def test_add_edge_that_already_exists(self):
        graph = MinimalGraph()
        graph.add_node(1)
        graph.add_node(2)
        graph.add_edge(1, 2)

        with pytest.raises(ValueError):
            graph.add_edge(1, 2)


## Version 2 - Minimal Presentation With Edges Nested In Nodes


class NestedNodeGraph:
    """Undirected graph, allows loops, but no duplicate nodes or edges.

    The node identity is defined by the `key` attribute that must be hashable.
    Edges are stored inside the `Node` inner class as adjacency set, and are
    undirected.

    More relevant to applications where we care for edges only when given a
    specific node."""

    class Node:

        key: Hashable
        neighbours: set

        def __init__(self, key: Hashable):
            self.key = key
            self.neighbours = set([])

        def __hash__(self):
            return hash(self.key)

        def __str__(self):
            return f'Node(key={self.key}; neighbours={self.neighbours})'

        def add_neighbour(self, neighbour: 'Node') -> None:
            if neighbour in self.neighbours:
                raise ValueError(f'Neighbour {neighbour.key} of {self.key} already exists.')
            self.neighbours.add(neighbour)

    nodes: Set[Node]

    def __init__(self):
        self.nodes = set([])

    def __str__(self):
        nodes_str = ';'.join(map(lambda node: str(node.key), self.nodes))
        return f'NestedNodeGraph(nodes={nodes_str})'

    def add_node(self, key: Hashable) -> None:
        if key in (node.key for node in self.nodes):
            raise ValueError(f'Node {key} already exists.')
        self.nodes.add(self.Node(key))

    def add_edge(self, key_from: Hashable, key_to: Hashable) -> None:
        node_from, node_to = None, None

        for node in self.nodes:
            if node.key == key_from:
                node_from = node
            if node.key == key_to:
                node_to = node

        if node_from is None or node_to is None:
            raise ValueError(f'Edge ({key_from}, {key_to}) '
                             f'must be between nodes in graph.')

        node_from.add_neighbour(node_to)
        node_to.add_neighbour(node_from)


class TestNestedNodeGraph:
    
    @pytest.mark.parametrize(
        "nodes_list",
        [[(1, 2, 3, 4), ],
         [(1,), (2, 3), (4, 5, 6)],
         [(1, 2.0, "3", object, NestedNodeGraph, set), ]]
    )
    def test_add_node(self, nodes_list: List[Tuple]):
        graph = NestedNodeGraph()
        expected_nodes = []
    
        for nodes in nodes_list:
            for node in nodes:
                graph.add_node(node)
                expected_nodes.append(node)
            assert set(expected_nodes) == {node.key for node in graph.nodes}
    
    def test_add_node_that_already_exists(self):
        graph = NestedNodeGraph()
        graph.add_node(1)
        with pytest.raises(ValueError):
            graph.add_node(1)
    
    def test_add_edge(self):
        graph = NestedNodeGraph()
        for node in range(1, 4):
            graph.add_node(node)
        for edge in itertools.combinations(range(1, 4), 2):
            graph.add_edge(*edge)
    
    def test_add_edge_not_in_nodes(self):
        graph = NestedNodeGraph()
        graph.add_node(1)
        graph.add_node(2)
    
        for edge in ((1, 3), (3, 1), (3, 4)):
            with pytest.raises(ValueError):
                graph.add_edge(*edge)
    
    def test_add_edge_that_already_exists(self):
        graph = NestedNodeGraph()
        graph.add_node(1)
        graph.add_node(2)
        graph.add_edge(1, 2)
    
        with pytest.raises(ValueError):
            graph.add_edge(1, 2)
        with pytest.raises(ValueError):
            graph.add_edge(2, 1)


## Version 3 - Weighted Graph


class WeightedGraph:
    """Directed graph, loops allowed, weighted nodes and edges, no duplicate
    nodes or edges.

    Node identity is defined by `key` hashable attribute.
    Edge identity is defined by the hash of the tuple of two node keys, where
    the order matters.

    There are great uses for weighted graphs. Accessing the edges through the
    `edges` attribute only would be difficult when trying to traverse the
    graphs. But would be reasonable in other uses."""

    class Node:

        CT = TypeVar('CT', bound=Comparable)

        key: Hashable

        def __init__(self, key: Hashable, weight: Generic[CT]):
            self.key = key
            self.weight = weight

        def __hash__(self):
            return hash(self.key)

        def __str__(self):
            return f'Node(key={self.key})'

    class Edge:

        CT = TypeVar('CT', bound=Comparable)

        node_from: 'Node'
        node_to: 'Node'
        weight: Generic[CT]

        def __init__(
                self, node_from: 'Node', node_to: 'Node', weight: Generic[CT],
        ):
            self.node_from = node_from
            self.node_to = node_to
            self.weight = weight

        def __hash__(self):
            return hash((self.node_from.key, self.node_to.key))

        def __str__(self):
            return f'Edge(node_from={self.node_from.key}; ' \
                   f'node_to={self.node_to.key}; weight={self.weight})'
        
    nodes: Set[Node]
    edges: Set[Edge]
        
    def __init__(self):
        self.nodes = set([])
        self.edges = set([])
    
    def __str__(self):
        nodes_str = ';'.join(map(lambda node: str(node.key), self.nodes))
        edges_str = ';'.join(map(str, self.edges))
        return f'WeightedGraph(nodes={nodes_str}; edges={edges_str})'
    
    def add_node(self, key: Hashable, weight: Node.CT) -> None:
        if key in (node.key for node in self.nodes):
            raise ValueError(f'Node {key} already exists.')
        self.nodes.add(self.Node(key, weight))
    
    def add_edge(self, key_from: Hashable, key_to: Hashable, weight: Edge.CT):
        node_from, node_to = None, None

        for node in self.nodes:
            if node.key == key_from:
                node_from = node
            if node.key == key_to:
                node_to = node

        if node_from is None or node_to is None:
            raise ValueError(f'Edge ({key_from}, {key_to}) '
                             f'must be between nodes in graph.')

        if (key_from, key_to) in \
                ((edge.node_from, edge.node_to) for edge in self.edges):
            raise ValueError(f'Edge {(key_from, key_to)} already exists.')
        self.edges.add(self.Edge(node_from, node_to, weight))


# Question 3 - BFS


def bfs(graph: NestedNodeGraph, start_key: Hashable) -> Iterable[Comparable]:
    start_node = next(itertools.dropwhile(
        lambda node: node.key != start_key, graph.nodes))
    queue = [start_node]
    visited = {start_node}

    while len(queue) > 0:
        current_node = queue.pop(0)
        yield current_node.key
        for neighbour_node in current_node.neighbours:
            if neighbour_node not in visited:
                queue.append(neighbour_node)
                visited.add(neighbour_node)


@pytest.mark.parametrize(
    'edges,start_key,expected_bfs',
    [
        ([(1, 2), (1, 3), (1, 4),
         (2, 5), (2, 3),
         (3, 6),
         (4, 5), (4, 7),
         (6, 2)],
         1,
         (1, 2, 3, 4, 5, 6, 7)),

        ([(1, 2), (1, 3), (1, 4),
         (2, 5), (2, 3),
         (3, 6),
         (4, 5), (4, 7),
         (6, 2)],
         2,
         (2, 1, 3, 5, 6, 4, 7)),
    ]
)
def test_bfs(
        edges: List[Tuple[int, int]], start_key: int, expected_bfs: Tuple[int],
):
    graph = NestedNodeGraph()
    for key in range(1, 8):
        graph.add_node(key)
    for edge in edges:
        graph.add_edge(*edge)

    actual_bfs = tuple(bfs(graph, start_key))
    assert actual_bfs == expected_bfs
