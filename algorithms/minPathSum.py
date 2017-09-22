# -*- coding: utf-8 -*-

def min_path_sum(input_list):
    row_length = len(input_list)
    col_length = len(input_list[0])
    dp_list = [[] for i in range(row_length)]
    dp_list[0].append(input_list[0][0])

    for i in range(1, row_length):
        dp_list[0].append(input_list[0][i] + dp_list[0][i-1])
    for j in range(1, col_length):
        dp_list[j].append(input_list[j][0] + dp_list[j-1][0])

    for i in range(1, row_length):
        for j in range(1, col_length):
            dp_list[i].append(min(dp_list[i-1][j], dp_list[i][j-1]) + input_list[i][j])


    return dp_list

print min_path_sum([[1, 3, 5, 9], [8, 1, 3, 4], [5, 0, 6, 1], [8, 8, 4, 0]])
