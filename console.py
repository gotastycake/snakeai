from os import system


def update(field, steps, scores, function, directions):
    system('cls')
    s = ''
    for i in field:
        s += ' '.join([str(j) for j in i]) + '\n'
    s = s.replace('4', '╔')
    s = s.replace('5', '═')
    s = s.replace('6', '╗')
    s = s.replace('7', '║')
    s = s.replace('8', '╚')
    s = s.replace('9', '╝')
    s = s.replace('0', '▫')
    s = s.replace('1', '■')
    s = s.replace('2', '♥')
    print('{}\nSteps: {}\nScores: {}\nMethod: {}\nPredicted directions: {}'.format(s, steps, scores, function, [round(i, 3) for i in directions]))
