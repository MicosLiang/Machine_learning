import numpy as np
import pygame
import matplotlib.pyplot as plt

class Chess:
    def king(board, i, k):
        if board[i, k] > 0:
            bottom = 0
            top = 2
        else:
            bottom = 7
            top = 9
        left = 3
        right = 5
        action = []
        if i - left > 0:
            action.append([i - 1, k])
        if right - i > 0:
            action.append([i + 1, k])
        if k - bottom > 0:
            action.append([i, k-1])
        if top - k > 0:
            action.append([i, k+1])
        actions = []
        for each in action:
            if board[i, k] * board[each[0], each[1]] <= 0:
                actions.append(each)
        return actions

    def advisor(board, i, k):
        if board[i, k] > 0:
            bottom = 0
            top = 2
        else:
            bottom = 7
            top = 9
        left = 3
        right = 5
        action = []
        if i - left > 0 and right - i > 0:
            action.append([i - 1, k - 1])
            action.append([i - 1, k + 1])
            action.append([i + 1, k + 1])
            action.append([i + 1, k - 1])
        if top - k == 0:
            if right - i == 0:
                action.append([i - 1, k - 1])
            else:
                action.append([i + 1, k - 1])
        if k - bottom == 0:
            if right - i == 0:
                action.append([i - 1, k + 1])
            else:
                action.append([i + 1, k + 1])
        actions = []
        for each in action:
            if board[i, k] * board[each[0], each[1]] <= 0:
                actions.append(each)
        return actions

    def elephant(board, i, k):
        if board[i, k] > 0:
            bottom = 0
            top = 4
        else:
            bottom = 5
            top = 9
        left = 0
        right = 8
        action = []
        # can go to left
        left = i - left > 0
        right = right - i > 0
        down = k - bottom > 0
        up = top - k > 0
        if left:
            if down:
                if board[i - 1, k - 1] == 0:
                    action.append([i - 2, k - 2])
            if up:
                if board[i - 1, k + 1] == 0:
                    action.append([i - 2, k + 2])
        if right:
            if down:
                if board[i + 1, k - 1] == 0:
                    action.append([i + 2, k - 2])
            if up:
                if board[i + 1, k + 1] == 0:
                    action.append([i + 2, k + 2])    
        actions = []
        for each in action:
            if board[i, k] * board[each[0], each[1]] <= 0:
                actions.append(each)
        return actions

    def horse(board, i, k):
        bottom = 0
        top = 9
        left = 0
        right = 8
        action = []
        # can go to left
        left = i - left
        right = right - i
        down = k - bottom
        up = top - k
        if down > 1 and board[i, k - 1] == 0:  
            if left:
                action.append([i - 1, k - 2])
            if right:
                action.append([i + 1, k - 2])
        if up > 1 and board[i, k + 1] == 0:  
            if left:
                action.append([i - 1, k + 2])
            if right:
                action.append([i + 1, k + 2])
        if left > 1 and board[i - 1, k] == 0:  
            if up:
                action.append([i - 2, k + 1])
            if down:
                action.append([i - 2, k - 1])
        if right > 1 and board[i + 1, k] == 0:  
            if up:
                action.append([i + 2, k + 1])
            if down:
                action.append([i + 2, k - 1])
        actions = []
        for each in action:
            if board[i, k] * board[each[0], each[1]] <= 0:
                actions.append(each)
        return actions

    def rook(board, i, k):
        bottom = 0
        top = 9
        left = 0
        right = 8
        action = []
        # can go to left
        left = i - left
        right = right - i
        down = k - bottom
        up = top - k
        for t in range(1, left+1):
            if board[i - t, k] == 0:
                action.append([i-t, k])
            else:
                if board[i, k] * board[i -t, k] < 0:
                    action.append([i-t, k])
                break
        for t in range(1, right+1):
            if board[i + t, k] == 0:
                action.append([i+t, k])
            else:
                if board[i, k] * board[i + t, k] < 0:
                    action.append([i + t, k])
                break
        for t in range(1, up + 1):
            if board[i, k + t] == 0:
                action.append([i, k + t])
            else:
                if board[i, k] * board[i, k + t] < 0:
                    action.append([i, k + t])
                break
        for t in range(1, down + 1):
            if board[i, k - t] == 0:
                action.append([i, k - t])
            else:
                if board[i, k] * board[i, k - t] < 0:
                    action.append([i, k - t])
                break
        return action

    def cannon(board, i, k):
        bottom = 0
        top = 9
        left = 0
        right = 8

        action = []

        left = i - left
        right = right - i
        down = k - bottom
        up = top - k

        no_block = True
        for t in range(1, left+1):
            if no_block and board[i - t, k] == 0:
                    action.append([i-t, k])
            else:
                if no_block:
                    no_block = False
                else:
                    if board[i, k] * board[i-t, k] < 0:
                        action.append([i-t, k])
                        break
        no_block = True
        for t in range(1, right+1):
            if no_block and board[i + t, k] == 0:
                action.append([i+t, k])
            else:
                if no_block:
                    no_block = False
                else:
                    if board[i, k] * board[i+t, k] < 0:
                        action.append([i+t, k])
                        break
        no_block = True
        for t in range(1, up + 1):
            if no_block and board[i, k + t] == 0:
                action.append([i, k + t])
            else:
                if no_block:
                    no_block = False
                else:
                    if board[i, k] * board[i, k + t] < 0:
                        action.append([i, k + t])
                        break
        no_block = True
        for t in range(1, down + 1):
            if no_block and board[i, k - t] == 0:
                action.append([i, k - t])
            else:
                if no_block:
                    no_block = False
                else:
                    if board[i, k] * board[i, k - t] < 0:
                        action.append([i, k - t])
                        break
        return action

    def pawn(board, i, k):
        left = 8
        right = 0
        if board[i, k] > 0:
            bottom = 9
            top = 9
            if k > 4:
                left = 0
                right = 8
        else:
            bottom = 0
            top = 0
            if k < 5:
                left = 0
                right = 8
        action = []
        if i - left > 0:
            action.append([i - 1, k])
        if right - i > 0:
            action.append([i + 1, k])
        if k - bottom > 0:
            action.append([i, k-1])
        if top - k > 0:
            action.append([i, k+1])

        actions = []
        for each in action:
            if board[i, k] * board[each[0], each[1]] <= 0:
                actions.append(each)
        return actions

    def action(defs, board, i, k):
        _type = int(np.abs(board[i, k]) - 1)
        return defs[_type](board, i, k)

class Game:
    def evaluate(board, agent, w):
        # if red[0].shape[0] and greeen[0].shape[0]:
        #     if red[0][0] == greeen[0][0]:
        #         if np.sum(board[red[0][0], red[1][0]:greeen[1][0]]) == 1:
        #             return -10000 * (1 if who == agent else -1)
        qizi = board * agent
        mark = np.sum(np.bincount(qizi.flatten() + 7) * w)
        return mark

    def action(board, who):
        boards = []
        defs = [Chess.king, Chess.advisor, Chess.elephant, Chess.horse, Chess.rook, Chess.cannon, Chess.pawn]
        for i in range(9):
            for k in range(10):
                if board[i, k] * who > 0:
                    actions = Chess.action(defs, board, i, k)
                    for action in actions:
                        _board = board.copy()
                        _board[action[0], action[1]] = _board[i, k]
                        _board[i, k] = 0
                        boards.append(_board)
        return boards

    def search(board, agent, who, w, alpha = -np.inf, beta = np.inf, deep = 4):
        if deep:
            nexts = Game.action(board, who)
            cnt = 0
            i = 0
            second = alpha != -np.inf
            for _next in nexts:
                value, tmp = Game.search(_next, agent, who * -1, w, alpha, beta, deep - 1)
                if agent == who:
                    if value > alpha:
                        alpha = value
                        i = cnt
                else:
                    if value < beta:
                        beta = value
                        i = cnt
                cnt += 1
                if alpha >= beta:
                    break
            if agent == who:
                return alpha, nexts[i]
            return beta, nexts[i]
        else:
            return Game.evaluate(board, agent, w), None

    def new_board():
        board = np.zeros([9,10],dtype="int")
        board[:, 0] = np.array([5, 4, 3, 2, 1, 2, 3, 4, 5])
        board[1, 2] = 6
        board[7, 2] = 6
        board[:, 3] = np.array([7, 0, 7, 0, 7, 0, 7, 0, 7])
        board[:, 6] = np.array([7, 0, 7, 0, 7, 0, 7, 0, 7]) * -1
        board[1, 7] = -6
        board[7, 7] = -6
        board[:, 9] = np.array([5, 4, 3, 2, 1, 2, 3, 4, 5]) * -1
        return board

    def monte(board, who):
        cnt = 0
        _who = who
        while True:
            actions = Game.action(board, who)
            id = np.random.choice(range(len(actions)), 1)[0]
            board = actions[id]
            if not (1 in board):
                return -1
            elif not (-1 in board):
                return 1
            who = who * -1
            cnt += 1
            if cnt > 300:
                return 0

from qipu import Charscore
class Game2(Game):
    def evaluate(board, agent):
        if agent.who in board and -agent.who in board:
            return agent.charscore.evaluate(board)
        else:
            return (1 if agent.who in board else -1) * np.inf

    def search(board, agent, who, w, alpha = -np.inf, beta = np.inf, deep = 4):
        if deep:
            nexts = Game2.action(board, who)
            cnt = 0
            i = 0
            # if len(nexts) > 5:
            #     nexts_value = np.array([Game2.evaluate(board, agent) for board in nexts])
            #     nexts_value_indexs = nexts_value.argsort()[-5:]
            #     _nexts = []
            #     for _index in nexts_value_indexs:
            #         _nexts.append(nexts[_index])
            #     nexts = _nexts
            for _next in nexts:
                value, tmp = Game2.search(_next, agent, who * -1, w, alpha, beta, deep - 1)
                if agent.who == who:
                    if value > alpha:
                        alpha = value
                        i = cnt
                else:
                    if value < beta:
                        beta = value
                        i = cnt
                cnt += 1
                if alpha >= beta:
                    break
            if agent.who == who:
                return alpha, nexts[i]
            return beta, nexts[i]
        else:
            return Game2.evaluate(board, agent), None

            
class Agent:
    def __init__(self,w, who = 1, deep = 4) -> None:
        self.w = w
        self.who = who
        self.deep = deep
        self.charscore = Charscore(who)
        pass

    def move(self, board):
        tmp, board = Game2.search(board, self, self.who, self.w, deep = self.deep)
        return board

class Competition:
    def __init__(self, agent_red, agent_green, board = None) -> None:
        if np.all(board == None):
            self.board = Game.new_board()
        else:
            self.board = board
        self.who = 0
        self.agents = [agent_red, agent_green]
        self.history_r = []
        self.history_g = []
        pass

    def judge(self):
        if not (1 in self.board):
            return -1
        elif not (-1 in self.board):
            return 1
        return 0

    def update(self):
        self.board = self.agents[self.who].move(self.board)
        self.who = 1 - self.who
        return self.judge()
    
    def run(self):
        ans = 0
        while ans == 0:
            self.history_r.append(Game2.evaluate(self.board, agent1))
            self.history_g.append(Game2.evaluate(self.board, agent2))
            ans = self.update()
            print(self.board)
        plt.plot(self.history_r, color="red")
        plt.plot(self.history_g, color="green")
        plt.show()
        return ans

class Player():
    def __init__(self, agent) -> None:
        self.board = Game.new_board()
        self.agent = agent
        if self.agent.who == 1:
            self.board = self.agent.move(self.board)
        print(self.board)
        pass

    def move(self, i, k, _i, _k):
        self.board[_i, _k] = self.board[i, k]
        self.board[i, k] = 0
        print(self.board)
        self.board = self.agent.move(self.board)
        print(self.board)

class GUI:
    def __init__(self) -> None:
        # 设置主屏幕大小
        size = (500, 450)
        self.screen = pygame.display.set_mode(size)
        # 设置一个控制主循环的变量
        done = False
        #创建时钟对象
        clock = pygame.time.Clock()
        while not done:
            # 设置游戏的fps
            clock.tick(10)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True  # 若检测到关闭窗口，则将done置为True

        pygame.draw.lines(self.screen, [255, 0, 0], [10,10], [100, 180])
        pass


import time
if __name__ == "__main__":
    # pygame.init()
    default = np.array([-80, -300, -500, -300, -250, -250, -1000, 0, 1000, 250, 250, 300, 500, 300, 80])
    # agent1 = Agent(w = default, who = 1, deep=3)
    agent2 = Agent(w = default, who = -1, deep=3)
    # agent1.charscore = red_char
    agent2.charscore = green_char
    # competition = Competition(agent1, agent2)
    # competition.run()
    player = Player(agent2)
