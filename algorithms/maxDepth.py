# -*- coding: utf-8 -*-
'''
给定一个二叉树，找出其最大深度。
二叉树的深度为根节点到最远叶子节点的距离。
'''

'''
首先判断根节点是否为空，如果为空，则返回0。再使用递归进行求解
>>>>>>> origin/master
'''

class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None

def maxDepth(root):
    # write your code here
    if root == None:
        return 0
    leftDepth = maxDepth(root.left)
    rightDepth = maxDepth(root.right)
    if leftDepth > rightDepth:
        return leftDepth + 1
    else:
        return rightDepth + 1
