# -*- coding: utf-8 -*-

'''
给定两个字符串，请设计一个方法来判定其中一个字符串是否为另一个字符串的置换。
置换的意思是，通过改变顺序可以使得两个字符串相等。

"abc" 为 "cba" 的置换。
"aabc" 不是 "abcc" 的置换。
'''

'''
这题主要是判断A里的字符串和B中出现的字符串是否一毛一样，
最简单可以使用两个for循环，这样耗时会很多。
可以遍历A中的所有字符串，计算hash值的和，同样计算B的hash值，判断两个hash值是否相同。
'''

def Permutation(A, B):
    # write your code here
    sum_a = 0
    sum_b = 0
    for i in A:
        sum_a += hash(i)
    for j in B:
        sum_b += hash(j)
    if sum_a == sum_b:
        isTrue = True
    else:
        isTrue = False
    return isTrue



