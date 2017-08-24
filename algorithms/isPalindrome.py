# -*- coding: utf-8 -*-
'''
判断一个正整数是不是回文数。
回文数的定义是，将这个数反转之后，得到的数仍然是同一个数。

11, 121, 1, 12321 这些是回文数。
23, 32, 1232 这些不是回文数。
'''

'''
回文序列 定义两个指针，一个从结尾来扫，一个从开头来扫
'''
def isPalindrome(num):
    num_str = str(num)
    i = len(num_str)
    j = 0
    isTrue = True
    while i > j:
        if num_str[i] == num_str[j]:
            i = i - 1
            j = j + 1
            continue
        else:
            isTrue = False
            break
    return isTrue