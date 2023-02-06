import re
import numpy as np
# from minmax import Game
import matplotlib.pyplot as plt

def to_number(type):
    if type == "车":
        return 5
    elif type == "炮":
        return 6
    elif type == "马":
        return 4
    elif type == "相" or type == "象":
        return 3
    elif type == "士" or type == "仕":
        return 2
    elif type == "帅" or type == "将":
        return 1
    elif type == "兵" or type == "卒":
        return 7
    return 0

def to_method(type):
    if type == "进":
        return 1
    elif type == "退":
        return -1
    elif type == "平":
        return 2
    return 0

def hanzi_to_number(type):
    if type == "一":
        return 1
    elif type == "二":
        return 2
    elif type == "三":
        return 3
    elif type == "四":
        return 4
    elif type == "五":
        return 5
    elif type == "六":
        return 6
    elif type == "七":
        return 7
    elif type == "八":
        return 8
    elif type == "九":
        return 9
    elif type == "前":
        return 10
    elif type == "后":
        return 11
    return 0

def array_tostr(arr):
    return [str(each) for each in arr]

def get_char(board):
    nns = []
    for i in range(9):
        line = board[i,:]
        line = line[line != 0]
        if len(line) > 1:
            for n in range(1,len(line)):
                nn = line[n-1:n+1]
                nn = ''.join(array_tostr(nn))
                nns.append(nn)
        if len(line) > 2:
            for n in range(2,len(line)):
                nn = line[n-2:n+1]
                nn = ''.join(array_tostr(nn))
                nns.append(nn)
        if len(line) > 3:
            for n in range(3,len(line)):
                nn = line[n-3:n+1]
                nn = ''.join(array_tostr(nn))
                nns.append(nn)
        for n in range(len(line)):
            nns.append(str(line[n]))
    return nns

class Charscore:
    def __init__(self, type) -> None:
        self.type = type
        self.scores = {}
        pass

    def add(self, char, val):
        try:
            self.scores[char] += val
        except Exception as e:
            self.scores[char] = val

    def get(self, char):
        try:
            return self.scores[char]
        except Exception as e:
            return 0

    def evaluate(self, board):
        val = 0
        _chars = get_char(board)
        for char in _chars:
            val += self.get(char)
        return val
    

def move(board, obj, pos, method, aim, who):
    who = int(who)
    if pos <= 9:
        if who == 1:
            pos = 9 - pos
        else:
            pos = pos - 1
    if obj == 1:
        _pos = np.where(board == obj*who)
        i = _pos[0][0]
        k = _pos[1][0]
        if method == 2:
            if who == 1:
                aim = 9 - aim
            else:
                aim = aim - 1
            board[aim, k] = board[i,k]
            board[i, k] = 0
        else:
            board[i, k + aim*who*method] = board[i, k]
            board[i, k] = 0
    elif obj == 2:
        objs = np.where(board[pos,:] == int(obj*who))[0]
        if len(objs) == 2:
            if method == 1:
                k = min(objs * who) * who
            else:
                k = max(objs * who) * who
        else:
            k = objs[0]
        if who == 1:
            aim = 9 - aim
        else:
            aim = aim - 1
        board[aim, k + 1*who*method] = board[pos, k]
        board[pos, k] = 0
        i = pos
    elif obj == 3:
        objs = np.where(board[pos,:] == int(obj*who))[0]
        if len(objs) == 2:
            if method == 1:
                k = min(objs * who) * who
            else:
                k = max(objs * who) * who
        else:
            k = objs[0]
        if who == 1:
            aim = 9 - aim
        else:
            aim = aim - 1
        board[aim, k + 2*who*method] = board[pos, k]
        board[pos, k] = 0
        i = pos
    elif obj == 4:
        if pos == 10 or pos == 11:
            if pos == 10:
                indexs = np.arange(0,10,1)
                if who == 1:
                    indexs = 9 - indexs
            else:
                indexs = np.arange(0,10,1)
                if who == -1:
                    indexs = 9 - indexs    
            for index in indexs:
                if int(obj*who) in board[:,index]:
                    i = np.where(board[:,index] == int(obj*who))[0][0]
                    k = index
                    break
        else:
            i = pos
            k = np.where(board[i,:] == int(obj*who))[0][0]
        if who == 1:
            aim = 9 - aim
        else:
            aim = aim - 1
        if np.abs(aim - i) == 2:
            board[aim, k + 1*who*method] = board[i, k]
        else:
            board[aim, k + 2*who*method] = board[i, k]
        board[i, k] = 0
    elif obj == 5 or obj == 6 or obj == 7:
        if pos == 10 or pos == 11:
            if pos == 10:
                indexs = np.arange(0,10,1)
                if who == 1:
                    indexs = 9 - indexs
            else:
                indexs = np.arange(0,10,1)
                if who == -1:
                    indexs = 9 - indexs    
            for index in indexs:
                if int(obj*who) in board[:,index]:
                    _is = np.where(board[:,index] == int(obj*who))[0]
                    if len(_is) == 1 and obj != 7:
                        i = _is[0]
                    else:
                        i = -1
                        for _i in _is:
                            if len(np.where(board[_i,:] == int(obj*who))[0]) == 2:
                                i = _i
                                break
                        if i < 0:
                            continue
                    k = index
                    break
        else:
            i = pos
            k = np.where(board[i,:] == int(obj*who))[0][0]
        if method == 2:
            if who == 1:
                aim = 9 - aim
            else:
                aim = aim - 1
            board[aim, k] = board[i, k]
        else:
            board[i, k + aim*who*method] = board[i, k]
        board[i, k] = 0
    return board

if __name__ == "__main__":
    txt = []
    with open("res.txt") as f:
        txt = f.readlines()
    qiju = []
    finals = []
    states = []
    for each in txt:
        each = each.strip()
        if each[0] == "(":
            continue
        final = each[-4:]
        if final[1] == "红":
            final = 1
        elif final[1] == "黑":
            final = -1
        elif final[1] == "和":
            final = 0
        else:
            print(final)
            continue
        each = each[:-4]
        steps = []
        _steps = each.split(".")[1:]
        _board = Game.new_board()
        steps.append(_board)
        for step in _steps:
            board = _board.copy()

            step1 = step[:4]
            if step1[0] == "前" or step1[0] == "后":
                obj = to_number(step1[1])
                pos = hanzi_to_number(step1[0])
            else:
                obj = to_number(step1[0])
                pos = hanzi_to_number(step1[1])
            method = to_method(step1[2])
            aim = hanzi_to_number(step1[3])
            if obj and pos >= 0 and method and aim:
                board = move(board, obj, pos, method, aim, 1)
            steps.append(board.copy())

            if len(step) > 7:
                step2 = step[4:8]
                if step2[0] == "前" or step2[0] == "后":
                    obj = to_number(step2[1])
                    pos = 10 if step2[0] == "前" else 11
                else:
                    obj = to_number(step2[0])
                    pos = int(step2[1])
                method = to_method(step2[2])
                aim = int(step2[3])
                if obj and pos >= 0 and method and aim:
                    board = move(board, obj, pos, method, aim, -1)
                steps.append(board)

            _board = board
        states.append(steps)
        finals.append(final)

    red_char = Charscore(1)
    green_char = Charscore(-1)
    he_char = Charscore(0)
    fix_1 = 1 * np.sum(np.array(finals) == -1) / np.sum(np.array(finals) == 1)
    fix_2 = -1 * np.sum(np.array(finals) == 1) / np.sum(np.array(finals) == 1)
    for i in range(len(finals)):
        final = finals[i]
        state = states[i]
        for board in state:
            _chars = get_char(board)
            for char in _chars:
                if final:
                    red_char.add(char, fix_1*final)
                    green_char.add(char, fix_2*final)
                    
    # red_wins_y = []
    # green_wins_y = []
    # he_y = []
    # for i in range(len(finals)):
    #     _y = []
    #     state = states[i]
    #     for board in state:
    #         val = 0
    #         _chars = get_char(board)
    #         for char in _chars:
    #             val += red_char.get(char)
    #         _y.append(val)
    #     if finals[i] == 1:
    #         red_wins_y.append(_y)
    #     elif finals[i] == -1:
    #         green_wins_y.append(_y)
    #     else:
    #         he_y.append(_y)
    # for each in red_wins_y:
    #     plt.plot(each, color = "red")
    # for each in green_wins_y:
    #     plt.plot(each, color = "green")
    # for each in he_y:
    #     plt.plot(each, color = "blue")
    # plt.show()



    _x = np.zeros([len(finals),90])
    _y = np.zeros([len(finals),3])
    for i in range(len(finals)):
        final = finals[i]
        state = states[i]
        board = int(np.random.choice(np.arange(0,len(state),1), size=1)[0])
        board = state[board].flatten()
        _x[i,:] = board
        _y[i,final+1] = 1
    
    model = Dense(_x, _y, loss='cross_entropy')
    model.add(512)
    model.add(512)
    model.add(512,activate = "tanh")
    model.add(3, activate = "softmax")
    model.init()
    model.train(epoch=1)