# The main file of SnakeAI
from random import randint, choice
from msvcrt import kbhit, getch

from time import sleep

from snake import Snake
from simple_snake_moving import *
import console

num_snake = 1
num_food = 2
num_field = 0

keys = {
    b's': 'down',
    b'w': 'up',
    b'a': 'left',
    b'd': 'right',
}

file_settings = 'settings.txt'
settings = {}

state = 0

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
    field.append([4]+[5]*field_size[1]+[6])

    for i in range(field_size[0]):
        field.append([7]+[num_field]*field_size[1]+[7])

    field.append([8]+[5]*field_size[1]+[9])


def generate_food():
    coords = (randint(1, field_size[0]), randint(1, field_size[1]))
    while field[coords[0]][coords[1]] in [1,2]:
        coords = (randint(1, field_size[0]), randint(1, field_size[1]))
    return coords


def check_direction(old, new):
    if not (old == 'up' and new == 'down' or
            old == 'down' and new == 'up' or
            old == 'left' and new == 'right' or
            old == 'right' and new == 'left'):
            return new
    return old


if __name__ == '__main__':
    load_settings()
    setup()

    game_is_on = True
    direction = 'right'

    food_coords = generate_food()
    field[food_coords[0]][food_coords[1]] = num_food

    snake = Snake()
    snake.create(field_size)
    field[snake.body[0][0]][snake.body[0][1]] = num_snake
    field[snake.body[-1][0]][snake.body[-1][1]] = num_snake

    while game_is_on:
        tail_coords = snake.move(direction, food_coords)
        if snake.body[-1] == food_coords:
            scores += 1
            food_coords = generate_food()
            field[food_coords[0]][food_coords[1]] = num_food
        else:
            field[tail_coords[0]][tail_coords[1]] = num_field
        field[snake.body[-1][0]][snake.body[-1][1]] = num_snake

        steps += 1

        # check if loss
        if snake.hit_wall(field_size):
            game_is_on = False
            state = 0
        if snake.hit_itself():
            game_is_on = False
            state = 1

        # # choosing a direction
        # k = getch()
        # # print(k)
        # key = keys.get(k, direction)
        # if not (direction == 'up' and key == 'down' or
        #     direction == 'down' and key == 'up' or
        #     direction == 'left' and key == 'right' or
        #     direction == 'right' and key == 'left'):
        #     direction = key

        # generating "stupid direction"
        #new_direction = generate_stupid_direction(field_size, snake.body[-1])

        # generating "moving on food"
        new_direction = generate_move_on_food(food_coords, snake.body[-1])
        direction = check_direction(direction, new_direction)
        console.update(field, steps, scores, generate_move_on_food.__name__)

        sleep(0.2)
    states = [
    'Snake hit the wall.',
    'Snake hit itself.',
    ]
    print("Game over! {}".format(states[state]))
