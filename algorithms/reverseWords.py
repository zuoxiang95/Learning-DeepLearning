# -*- coding: utf-8 -*-
'''
将一个整数中的数字进行颠倒，当颠倒后的整数溢出时，返回 0 (标记为 32 位整数)。
给定 x = 123，返回 321
给定 x = -123，返回 -321
'''

'''
判断数字是否大于为32位整数，如果是，则返回0，不是，遍历数字的每一位，定义一个变量储存正负值，反转数字。
'''


def reverseInteger(n):
    # write your code here
    if (n > 2 ** 31 - 1) or (n < -2 ** 31):
        return 0
    str_n = str(n)
    i = len(str(n)) - 1
    result = []
    is_negative = 1
    while i >= 0:
        if str_n[i] == '-':
            is_negative = -1
            i = i - 1
            continue
        else:
            result.append(str_n[i])
            i = i - 1
    tmp = int(''.join(result)) * is_negative
    if (tmp > 2 ** 31 - 1) or (tmp < -2 ** 31):
        return 0
    else:
        return tmp
