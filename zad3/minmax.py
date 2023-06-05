from board import Board

Heuristics = [[3, 2, 3], [2, 4, 2], [3, 2, 3]]

Heuristics5 = [[2, 2, 1, 2, 2], [2, 3, 3, 3, 2], [2, 3, 4, 3, 2], [2, 3, 3, 3, 2], [3, 2, 1, 2, 3]]

def minmax(board, depth, turn):
    if board.gameOver() or depth == 0:
        return rate(board)

    moves = board.getBoardsList(turn)

    if turn == 'o':
        best = -100
        oldBest = best
        for move in moves:
            best = max(best, minmax(move, depth-1, 'x'))
            if oldBest < best:
                oldBest = best
                position =  move.getMove()
        board.makeMove(position, turn)
        return best

    if turn == 'x':
        best = 100
        oldBest = best
        for move in moves:
            best = min(best, minmax(move, depth-1, 'o'))
            if oldBest > best:
                oldBest = best
                position =  move.getMove()
        board.makeMove(position, turn)
        return best

def rate(board):
    result = board.checkWhoWon()
    if result == 'o': return 10
    elif result == 'x': return -10
    elif result == 'd': return 0
    else: return Heuristics[board.getMove()[0]] [board.getMove()[1]]


def start(size, depth):
    board = Board(size)
    move = 1
    while (not board.gameOver()):
        if move%2 == 1:
            turn = 'o'
        else:
            turn = 'x'

        print(f"Move: {move} - {turn}", "Best: ",minmax(board, depth, turn))
        board.display()
        print("Who wins? ->", board.checkWhoWon())
        move += 1


def main():
    start(size = 3, depth = 9)

if __name__ == '__main__':
    main()
