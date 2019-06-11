# The main file of SnakeAI
from snake import Snake

file_settings = 'settings.txt'
settings = {}

field_size = ()

field = []

scores = 0
steps = 0


def load_settings():
    with open(file_settings, 'r') as f:
        for line in f:
            split_line = line.split()
            settings[split_line[0]] = split_line[1]


def setup():
    # setting field size
    global field_size
    field_size = (int(settings['field_size_x']), int(settings['field_size_y']))

    # setting field
    field.append([-1]*field_size[1])

    for i in range(field_size[0]):
        field.append([-1]+[0]*field_size[1]+[-1])

    field.append([-1] * field_size[1])


if __name__ == '__main__':
    load_settings()
    setup()

    game_is_on = True
    direction = 'right'

    snake = Snake()
    snake.create()

    while game_is_on:
        snake.move(direction)
        scores += 1
        steps += 1

        # check if loss
        if snake.hit_wall() or snake.hit_itself():
            game_is_on = False

