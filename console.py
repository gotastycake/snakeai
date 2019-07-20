from os import system


def update(field, scores, lifetime, steps, directions, X, epoch, cur_dir):
    X = X[0]
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
    s += 'Epoch: {}\n'.format(epoch)
    s += 'Scores: {}\n'.format(scores)
    s += 'Lifetime: {}\n'.format(lifetime)
    s += 'Steps: {}\n'.format(steps)
    s += 'Predicted direction: {}\n'.format(cur_dir)
    s += '{}\n'.format([round(i, 3) for i in directions])
    s += 'X:           w     t      f\n'
    s += 'up        : {:<6.3} {:<6.3} {:<6.3}\n'.format(X[0], X[1], X[2])
    s += 'up-right  : {:<6.3} {:<6.3} {:<6.3}\n'.format(X[3], X[4], X[5])
    s += 'right     : {:<6.3} {:<6.3} {:<6.3}\n'.format(X[6], X[7], X[8])
    s += 'down-right: {:<6.3} {:<6.3} {:<6.3}\n'.format(X[9], X[10], X[11])
    s += 'down      : {:<6.3} {:<6.3} {:<6.3}\n'.format(X[12], X[13], X[14])
    s += 'down-left : {:<6.3} {:<6.3} {:<6.3}\n'.format(X[15], X[16], X[17])
    s += 'left      : {:<6.3} {:<6.3} {:<6.3}\n'.format(X[18], X[19], X[20])
    s += 'up-left   : {:<6.3} {:<6.3} {:<6.3}\n'.format(X[21], X[22], X[23])
    print(s)
