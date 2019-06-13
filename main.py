# The main file of SnakeAI

from time import sleep

from snake import Snake
from simple_snake_moving import *
from config import *
import console

# settings
settings = {}       # dict of settings imported from "settings" file
delay = 0           # delay between frames

# field variables
field_size = ()
field = []


# loads settings from file
def load_settings():
    with open(file_settings, 'r') as f:
        for line in f:
            split_line = line.split()
            settings[split_line[0]] = split_line[1]

    # setting field size
    global field_size
    field_size = (int(settings['field_size_x']), int(settings['field_size_y']))

    # setting delay
    global delay
    delay = float(settings['delay'])


# sets initial state for game
def setup():
    global field
    field = list()

    # setting field
    field.append([4]+[5]*field_size[1]+[6])

    for i in range(field_size[0]):
        field.append([7]+[num_field]*field_size[1]+[7])

    field.append([8]+[5]*field_size[1]+[9])


# generates food coordinates
def generate_food():
    coords = (randint(1, field_size[0]), randint(1, field_size[1]))
    while field[coords[0]][coords[1]] in [1, 2]:
        coords = (randint(1, field_size[0]), randint(1, field_size[1]))
    return coords


# checks if a snake can move in a new direction
def check_direction(old, new):
    if not (old == 'up' and new == 'down' or
            old == 'down' and new == 'up' or
            old == 'left' and new == 'right' or
            old == 'right' and new == 'left'):
        return new
    return old


def play_game(decision_function=None):
    setup()

    # game variables
    game_is_on = True
    scores = 0
    steps = 0
    direction = 'right'
    state = ''  # describes the reason of game over

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
            state = 'Snake hit the wall.'
        if snake.hit_itself():
            game_is_on = False
            state = 'Snake hit itself.'

        if not decision_function:
            new_direction = generate_move_on_food(food_coords, snake.body[-1])
        else:
            new_direction = decision_function()
        direction = check_direction(direction, new_direction)

        #console.update(field, steps, scores, decision_function.__name__)
        console.update(field, steps, scores, generate_move_on_food.__name__)

        sleep(delay)

    print("Game over! {}".format(state))

    return scores


if __name__ == '__main__':
    load_settings()

    while True:
        score = play_game()
        print('Score: {}'.format(score))
        input()
