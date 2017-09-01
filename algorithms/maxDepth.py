# -*- coding: utf-8 -*-
'''
<<<<<<< HEAD
将一个整数中的数字进行颠倒，当颠倒后的整数溢出时，返回 0 (标记为 32 位整数)。
给定 x = 123，返回 321
给定 x = -123，返回 -321
'''

'''
判断数字是否大于为32位整数，如果是，则返回0，不是，遍历数字的每一位，定义一个变量储存正负值，反转数字。
=======
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
<<<<<<< HEAD
    pass
=======
    if root == None:
        return 0
    leftDepth = maxDepth(root.left)
    rightDepth = maxDepth(root.right)
    if leftDepth > rightDepth:
        return leftDepth + 1
    else:
        return rightDepth + 1
>>>>>>> origin/master
