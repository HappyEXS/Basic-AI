
class Board:
    def __init__(self, size, board=None):
        self.size = size
        self.lastMove = []
        if board:
            self.board = self.generateBoard(board)
        else:
            self.board = self.generateEmptyBoard()

    def __str__(self):
        return str(self.board)

    def getMove(self):
        return self.lastMove

    def makeMove(self, position, sign):
        self.board[position[0]][position[1]] = sign

    def gameOver(self):
        if self.checkWhoWon() is not None: return True
        else: return False

    def generateEmptyBoard(self):
        filler = None
        new_board = []
        for x in range(self.size):
            row = []
            for y in range(self.size):
                row.append(filler)
            new_board.append(row)
        return new_board

    def generateBoard(self, board):
        new_board = []
        for x in range(self.size):
            row = []
            for y in range(self.size):
                filler = board[x][y]
                row.append(filler)
            new_board.append(row)
        return new_board

    def display(self):
        print('-' * (self.size * 4 + 1))
        for row in self.board:
            line = '|'
            for sign in row:
                if sign is None: sign = ' '
                line += f' {sign} |'
            print(line)
            print('-' * (self.size * 4 + 1))

    def getBoardsList(self, sign):
        Moves = []
        for x in range(self.size):
            for y in range(self.size):
                if(self.board[x][y] is None):
                    tmp = Board(self.size, self.board)
                    tmp.board[x][y] = sign
                    tmp.lastMove = [x, y]
                    Moves.append(tmp)
        return Moves

    def getMovesList(self):
        Moves = []
        for x in range(self.size):
            for y in range(self.size):
                if(self.board[x][y] is None):
                    Moves.append([x, y])
        return Moves

    def checkWhoWon(self):
        # check horizontal
        for x in range(self.size):
            for y in range(1, self.size):
                if(self.board[x][y] == self.board[x][y-1] and self.board[x][y] is not None):
                    if y == self.size - 1:
                        return self.board[x][y]
                    else: continue
                else: break
        # check vertical
        for x in range(self.size):
            for y in range(1, self.size):
                if(self.board[y][x] == self.board[y-1][x] and self.board[y][x] is not None):
                    if y == self.size - 1:
                        return self.board[y][x]
                    else: continue
                else: break
        # check diagonal topleft-bottomright
        x = 1
        while(x < self.size):
            if(self.board[x][x] == self.board[x-1][x-1] and self.board[x][x] is not None):
                if x == self.size - 1:
                    return self.board[x][x]
                else: x += 1
            else: break
        # check diagonal topright-bottomleft
        x = 1
        y = self.size - 2
        while(x < self.size or y < self.size):
            if(self.board[x][y] == self.board[x-1][y+1] and self.board[x][y] is not None):
                if x == self.size - 1:
                    return self.board[x][y]
                else:
                    x += 1
                    y -= 1
            else: break
        # check draw
        for x in range(self.size):
            for y in range(self.size):
                if self.board[x][y] is None:
                    return None
                if x == self.size - 1 and  y == self.size - 1:
                    return 'd'

