import random
from typing import Generator, List


# Utilities


class TreeNode:

    def __init__(self, value, parent=None, left=None, right=None):
        self.value = value
        self.parent = parent
        self.left = left
        self.right = right

    def __hash__(self):
        return id(self)
        
    def __str__(self):
        return f"value={self.value}, " \
            f"parent={self.parent.value if self.parent else None} " \
            f"left={self.left.value if self.left else None}, " \
            f"right={self.right.value if self.right else None}"
        
    @classmethod
    def randomTreeNode(cls):
        return cls(random.randint(1, 20))
        
    @classmethod
    def randomTree(cls):
        
        tree_node = cls.randomTreeNode()
        stack = [tree_node]
        
        while stack:
            current_node = stack.pop()
            
            if random.choice([False, True]):
                current_node.left = cls.randomTreeNode()
                stack.append(current_node.left)
        
            if random.choice([False, True]):
                current_node.right = cls.randomTreeNode()
                stack.append(current_node.right)

            print(current_node)
                    
        return tree_node
    
    # Question 1
    
    @staticmethod
    def bothNonesOrBothNotNones(object1, object2) -> bool:
        return not ((object1 is None) ^ (object2 is None))
        
    def __eq__(self, other_tree):
        """Version 1: recursively implemented"""
        if self.value != other_tree.value:
            return False
        if not self.bothNonesOrBothNotNones(self.left, other_tree.left) or (self.left is not None and self.left != other_tree.left):
            return False
        if not self.bothNonesOrBothNotNones(self.right, other_tree.right) or (self.right is not None and self.right != other_tree.right):
            return False
        return True
        
    @staticmethod
    def isEquals(t1, t2) -> bool:
        """Version 2: iteratively implemented"""
        stack = [(t1, t2)]
        while stack:
            t1_node, t2_node = stack.pop()
            if (t1_node is None) ^ (t2_node is None):
                return False
            if t1_node is not None:
                if t1_node.value != t2_node.value:
                    return False
                stack.extend([(t1_node.left, t2_node.left), (t1_node.right, t2_node.right)])
        return True

    # Question 2

    @property
    def height(self) -> int:
        max_level = 0
        nodes_stack = [(self, 0)]
        
        while nodes_stack:
            current_node, current_level = nodes_stack.pop()
            if current_node.left is not None or current_node.right is not None:
                current_level += 1
                if current_level > max_level:
                    max_level = current_level
                if current_node.left is not None:
                    nodes_stack.append((current_node.left, current_level))
                if current_node.right is not None:
                    nodes_stack.append((current_node.right, current_level))
                    
        return max_level
    
    @property
    def height2(self) -> int:
        if self.left is None and self.right is None:
            return 0
        elif self.left is None:
            return self.right.height2 + 1
        elif self.right is None:
            return self.left.height2 + 1
        else:
            return max(self.left.height2, self.right.height2) + 1
    
    # Question 4
    
    def inorder(self) -> List:
        queue = [self]
        nodes = []
        seen = set([])
        
        while len(queue) > 0:
            curr_node = queue.pop()
            if curr_node in seen:
                nodes.append(curr_node)
                continue
            if curr_node.right is not None:
                queue.append(curr_node.right)
            queue.append(curr_node)
            seen.add(curr_node)
            if curr_node.left is not None:
                queue.append(curr_node.left)

        return nodes
    
    def inorder_generator(self) -> Generator:
        queue = [self]
        nodes = []
        seen = set([])
        
        while len(queue) > 0:
            curr_node = queue.pop()
            if curr_node in seen:
                yield curr_node
                continue
            if curr_node.right is not None:
                queue.append(curr_node.right)
            queue.append(curr_node)
            seen.add(curr_node)
            if curr_node.left is not None:
                queue.append(curr_node.left)

        return
    
    def inorder_recursive(self) -> List:
        nodes = []
        if self.left is not None:
            nodes.extend(self.left.inorder_recursive())
        nodes.append(self)
        if self.right is not None:
            nodes.extend(self.right.inorder_recursive())
        return nodes
    
    # Question 5
    
    def preorder(self) -> list:
        nodes = []
        stack = [self]
        
        while len(stack) > 0:
            curr_node = stack.pop()
            nodes.append(curr_node)
            if curr_node.right is not None:
                stack.append(curr_node.right)
            if curr_node.left is not None:
                stack.append(curr_node.left)
        
        return nodes
    
    def preorder_generator(self) -> list:
        stack = [self]
        
        while len(stack) > 0:
            curr_node = stack.pop()
            yield curr_node
            if curr_node.right is not None:
                stack.append(curr_node.right)
            if curr_node.left is not None:
                stack.append(curr_node.left)
    
    def preorder_recursive(self) -> list:
        nodes = [self]
        if self.left is not None:
            nodes.extend(self.left.preorder_recursive())
        if self.right is not None:
            nodes.extend(self.right.preorder_recursive())
        return nodes


def test_randomTree() -> None:
    print(TreeNode.randomTree())
    

# Question 1


def test_equalTrees() -> None:
    t1 = TreeNode.randomTree()
    t2 = TreeNode.randomTree()
    print(t1)
    print(t2)
    print(TreeNode.isEquals(t1, t1))
    print(t1 == t1)
    print(TreeNode.isEquals(t1, t2))
    print(t1 == t2)


# Question 2

    
def test_height() -> None:
    t = TreeNode.randomTree()
    print(t)
    assert t.height == t.height2
    
    
# Question 3


def removeIterative(root: TreeNode) -> None:
    if root is None: return
    stack = [root]
    
    while stack:
        current_node = stack.pop()
        if current_node.left is not None:
            stack.append(current_node.left)
            current_node.left = None
        if current_node.right is not None:
            stack.append(current_node.right)
            current_node.right = None


def removeRecursive(root: TreeNode) -> None:
    if root is None: return
    if root.left is not None:
        removeRecursive(root.left)
        root.left = None
    if root.right is not None:
        removeRecursive(root.right)
        root.right = None


def test_remove() -> None:
    t = TreeNode.randomTree()
    removeRecursive(t)


# Question 4


def test_inorder() -> None:
    t = TreeNode.randomTree()
    
    def print_result(func: str):
        print(','.join(map(lambda n: str(n.value), getattr(t, func)())))
    
    for func in ("inorder", "inorder_generator", "inorder_recursive"):
        print_result(func)


def test_preorder() -> None:
    t = TreeNode.randomTree()
    
    def print_result(func: str):
        print(','.join(map(lambda n: str(n.value), getattr(t, func)())))
    
    for func in ("preorder", "preorder_generator", "preorder_recursive"):
        print_result(func)


test_preorder()