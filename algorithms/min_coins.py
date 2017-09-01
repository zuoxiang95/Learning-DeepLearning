# -*- coding: utf-8 -*-

def min_coins(coins, amount):
    dp_list = [[] for i in coins]
    for i in dp_list:
        i.append(0)
    for i in range(1, amount):
        for j in range(len(coins)):

    print dp_list


min_coins([1,2,3], 5)