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