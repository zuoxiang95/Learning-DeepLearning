# -*- coding: utf-8 -*-
'''
给出一个包含大小写字母的字符串。求出由这些字母构成的最长的回文串的长度是多少。
数据是大小写敏感的，也就是说，"Aa" 并不会被认为是一个回文串。

给出 s = "abccccdd" 返回 7
一种可以构建出来的最长回文串方案是 "dccaccd"。
'''

'''
可以创建一个字典，字典中存放每个字母出现的个数。
将所有偶数的和加上所有奇数-1的和得到最长回文方案
'''
def longestPalindrome(s):
    # Write your code here
    my_dict = {}
    for i in s:
        if i not in my_dict:
            my_dict[i] = 1
        else:
            my_dict[i] += 1
    my_sum = 0
    print(my_dict['i'])
    for j in my_dict.values():
        if j % 2 == 0:
            my_sum += j
        else:
            continue
    return my_sum + 1
x = 'daslmASvAjSjoqj'
y = longestPalindrome(x)
print(y)