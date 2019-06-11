from os import system


def update(field, steps, scores, function):
    system('cls')
    for i in field:
        s = ' '.join([str(j) for j in i])
        s = s.replace('4', '╔')
        s = s.replace('5', '═')
        s = s.replace('6', '╗')
        s = s.replace('7', '║')
        s = s.replace('8', '╚')
        s = s.replace('9', '╝')
        s = s.replace('0', '▫')
        s = s.replace('1', '■')
        s = s.replace('2', '♥')
        print(s)
    print('Steps: {}\nScores: {}\nMethod: {}'.format(steps, scores, function))
