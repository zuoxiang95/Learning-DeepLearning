# -*- coding: utf-8 -*-
'''
给一个数组 nums 写一个函数将 0 移动到数组的最后面，非零元素保持原数组的顺序

 注意事项
1.必须在原数组上操作
2.最小化操作数

给出 nums = [0, 1, 0, 3, 12], 调用函数之后, nums = [1, 3, 12, 0, 0].
'''

'''
定义两个指针，一个从头开始扫，一个从尾扫，判断头指针指向的元素是否为0，
如果为0就从原来数组中删除掉0，并添加在最后，然后尾指针向前移动一位。
如果不为0，则头指针向后移动一位
'''

def moveZeroes(nums):
    i = len(nums)- 1
    while i > 0:
        if nums[i] == 0:
            nums.pop(i)
            nums.append(0)
        else:
            i = i - 1
    return nums


print(moveZeroes([1,0,2,0,0,3]))