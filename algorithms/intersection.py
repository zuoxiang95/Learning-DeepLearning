# -*- coding: utf-8 -*-
'''
计算两个数组的交

nums1 = [1, 2, 2, 1], nums2 = [2, 2], 返回 [2, 2].
'''

'''
要先对数组进行排序，排序后再定义两个指针，分别指向nums1和nums2，根据大小移动指针，相同则添加到结果的数组中。
'''
def intersection(nums1, nums2):
    nums1.sort()
    nums2.sort()
    len_i = len(nums1)
    len_j = len(nums2)
    i = 0
    j = 0
    result = []
    while (i < len_i) & (j < len_j):
        if nums1[i] == nums2[j]:
            result.append(nums1[i])
            i = i + 1
            j = j + 1
        elif nums1[i] < nums2[j]:
            i = i + 1
        else:
            j = j + 1
    return result
print(intersection([1,2,2,1], [2,2]))