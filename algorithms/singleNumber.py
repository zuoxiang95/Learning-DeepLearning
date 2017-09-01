# -*- coding: utf-8 -*-
'''
给出2*n + 1 个的数字，除其中一个数字之外其他每个数字均出现两次，找到这个数字。

给出 [1,2,2,1,3,4,3]，返回 4
'''

'''
遍历这个数组，使用位运算
'''

def longestPalindrome(s):
    my_sum = 0
    for i in s:
        my_sum = my_sum ^ i
    return my_sum

print longestPalindrome([1,1,2,2,3,3,4,5,5])