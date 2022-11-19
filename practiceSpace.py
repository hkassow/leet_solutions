
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
field1 =  [[0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0],
          [1, 1, 0, 1, 0],
          [1, 0, 1, 0, 1]]
figure1 = [[1, 1, 1],
          [1, 0, 1],
          [1, 0, 1]]

print(solve(field, figure))
print(solve(field1, figure1))