#collection of non-leetcode problems ive run across





# tetris blocks
# You are given a matrix of integers field of size height × width representing a game field, 
# and also a matrix of integers figure of size 3 × 3 representing a figure. Both matrices 
# contain only 0s and 1s, where 1 means that the cell is occupied, and 0 means that the cell is free.
# You choose a position at the top of the game field where you put the figure and then drop it down. 
# The figure falls down until it either reaches the ground (bottom of the field) or
# lands on an occupied cell, which blocks it from falling further. After the figure has stopped falling, some of the rows in the field may become fully occupied.

# inputs field m x n matrix and figure 3x3 matrix both consisting of 1's and 0's
# output position of figure such that when figure is dropped it fills up the row completely
# if we cannot find a position return -1
# note we cannot consider positions where any part of figure is outside the field even if that part of the figure holds only 0's


# solution breakdown =>
# 1) we iterate across all potential start positions
# 2) we keep dropping our figure until it reaches the bottom of field or it cannot be dropped anymore (moving down 1 row results in figure + field collision)
# 3) we check if the figure completes any rows => if true we return the start position
# 4) if not rows are filled with all possible start positions return -1
def solve(field, figure):
    def canFall(row, col):
        figurePoints = [[0,0], [1,0], [2,0], [1,1], [1,2], [2,2], [2,1], [0,2], [0,1]]
        if (row == len(field)): 
            return False
        for [x,y] in figurePoints:
            if figure[2-x][y] == 1:
                i,j = row-x, y+col
                if i < 0:
                    continue
                if field[i][j] == 1:
                    return False
        return True
    def isFilled(bottom, figurePos):
        for i in range(3):
            row = bottom - i
            if row < 0:
                return False
            filled = True
            for j in range(len(field[0])):
                col = j
                
                if field[row][j] == 0 and ( j-figurePos < 0 or 3 <= j-figurePos or figure[2-i][j-figurePos] == 0):
                    filled = False
                    break
            if filled:
                return True

    for j in range(len(field[0])- 2):
        i = 0

        while canFall(i+1, j):
            i += 1
        if isFilled(i, j):
            return j
    return -1


field = [[0, 0, 0, 0],
          [0, 0, 0, 0],
          [0, 0, 0, 0],
          [1, 0, 0, 1],
          [1, 1, 0, 1]]
figure = [[1, 1, 0],
          [1, 0, 0],
          [1, 0, 0]]

#solution will be -1 (note this has to do with figure needing to be completely inside matrix)
field1 =  [[0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0],
          [1, 1, 0, 1, 0],
          [1, 0, 1, 0, 1]]
figure1 = [[1, 1, 1],
          [1, 0, 1],
          [1, 0, 1]]

#solution will be 2


field2 = [[0, 0, 0],
         [0, 0, 0],
         [0, 0, 0],
         [1, 0, 0],
         [1, 1, 0]]
figure2 = [[0, 0, 1],
         [0, 1, 1],
         [0, 0, 1]]

#soltuion will be 0



# special dictionary
# problem:
# we want to design a dictionary that allows incrementing all the keys and values
# given a queryType ==> 'insert' || 'addToValue' || 'addToKey' || 'get'
# and a query ========>  [k,v]   ||   [v]        ||     [v]    ||   [k]
# return the value of all get operations
# solution: 
# using an offset for key and value we can achieve this in o(n) time without having to iterate through our dictionary every time
# when we insert we must subtract both the offsets to put the key in the correct spot and not double add our value offset
# when we get the key we must subtract the keyoffset to get the correct key and then add the value offset
# example:
# insert [5,10] => dict = {5:10}
# addToKey [10] => dict = {5:10}
# addToVal [5] => dict = {5:10}
# get [15] 15-10=5 => get dict[5] = 10 + 5 => res = 15
# insert [5,70] -5,65 => dict {5:10, -5:65}
# get 5 5-10=-5 => get dict[-5] = 65 + 5 => res = 15 + 70

def solution(queryType, query):
    key_offset = 0
    val_offset = 0
    
    store = {}
    
    res = 0
    
    for i in range(len(query)):
        code = queryType[i]
        q = query[i]
        
        if code == 'insert':
            store[q[0]-key_offset] = q[1] - val_offset
        elif code == 'addToValue':
            val_offset += q[0]
        elif code == 'addToKey':
            key_offset += q[0]
        else:
            if q[0]-key_offset in store:
                res += store[q[0]-key_offset] + val_offset
    return res




