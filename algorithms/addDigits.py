# -*- coding: utf-8 -*-
'''
给出一个非负整数 num，反复的将所有位上的数字相加，直到得到一个一位的整数。

给出 num = 38。
相加的过程如下：3 + 8 = 11，1 + 1 = 2。因为 2 只剩下一个数字，所以返回 2。
'''

'''
递归求和
'''
def addDigits(num):
    num_str = str(num)
    tmp = []
    for i in num_str:
        tmp.append(int(i))
    x = sum(tmp)
    if len(str(x)) != 1:
        return addDigits(x)
    else:
        return x

print(addDigits(38))

