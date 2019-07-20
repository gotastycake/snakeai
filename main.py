# The main file of SnakeAI

from os import system
from time import sleep

import numpy as np

from snake import Snake
from simple_snake_moving import *
from config import num_field, num_food, num_snake, directions, field_size, delay
import console
import train


max_steps = 50
n_train = 30
empty_field = []


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
def generate_food(field):
    coords = (randint(1, field_size[0]), randint(1, field_size[1]))
    while field[coords[0]][coords[1]] in [num_food, num_snake]:
        coords = (randint(1, field_size[0]), randint(1, field_size[1]))
    return coords


# checks if a snake can move in a new direction
def set_new_direction(old, pred):
    m = np.argmax(pred)
    new = directions[m]
    if not (old == 'up' and new == 'down' or
            old == 'down' and new == 'up' or
            old == 'left' and new == 'right' or
            old == 'right' and new == 'left'):
        return new
    return old


def play_game(model, epoch):
    setup()

    # game variables
    game_is_on = True
    score = 0
    steps = 0
    lifetime = 0
    direction = 'right'
    state = ''  # describes the reason of game over

    snake = Snake()
    snake.create(field_size)
    field[snake.body[0][0]][snake.body[0][1]] = num_snake
    field[snake.body[-1][0]][snake.body[-1][1]] = num_snake

    food_coords = generate_food(field)
    field[food_coords[0]][food_coords[1]] = num_food

    while game_is_on:
        X = train.count_X(field_size, field, snake, food_coords)
        pred = model.predict(X)

        direction = set_new_direction(direction, pred)

        tail_coords = snake.move(direction, food_coords)
        if snake.body[-1] == food_coords:
            score += 1
            food_coords = generate_food(field)
            field[food_coords[0]][food_coords[1]] = num_food
            steps = 0
        else:
            field[tail_coords[0]][tail_coords[1]] = num_field
        field[snake.body[-1][0]][snake.body[-1][1]] = num_snake

        steps += 1
        lifetime +=1

        # check if loss
        if snake.hit_wall(field_size):
            game_is_on = False
            state = 'Snake hit the wall.'
        if snake.hit_itself():
            game_is_on = False
            state = 'Snake hit itself.'
        if steps >= max_steps:
            game_is_on = False
            state = 'Snake lost its way.'
            lifetime = 0
        console.update(field, score, lifetime, steps, pred[0], X, epoch, direction)

        sleep(delay)

    print("Game over! {}".format(state))
    del snake
    return lifetime, score


def play_nogui(model):
    setup()

    # game variables
    game_is_on = True
    score = 0
    steps = 0
    lifetime = 0
    direction = 'right'
    state = ''  # describes the reason of game over

    snake = Snake()
    snake.create(field_size)
    field[snake.body[0][0]][snake.body[0][1]] = num_snake
    field[snake.body[-1][0]][snake.body[-1][1]] = num_snake

    food_coords = generate_food(field)
    field[food_coords[0]][food_coords[1]] = num_food

    while game_is_on:
        X = train.count_X(field_size, field, snake, food_coords)
        pred = model.predict(X)

        direction = set_new_direction(direction, pred)

        tail_coords = snake.move(direction, food_coords)
        if snake.body[-1] == food_coords:
            score += 1
            food_coords = generate_food(field)
            field[food_coords[0]][food_coords[1]] = num_food
            steps = 0
        else:
            field[tail_coords[0]][tail_coords[1]] = num_field
        field[snake.body[-1][0]][snake.body[-1][1]] = num_snake

        steps += 1
        lifetime +=1

        # check if loss
        if snake.hit_wall(field_size):
            game_is_on = False
        if snake.hit_itself():
            game_is_on = False
        if steps >= max_steps:
            game_is_on = False
            lifetime = 0
    del snake
    return lifetime, score


if __name__ == '__main__':
    system('cls')
    #event = input('train or play? ')
    event = 't'
    if event in ['train', 't']:
        train.train_models(n_train, load_models=False)
        train.df_stats.reset_index()
        train.df_stats.to_csv('output.csv')
    elif event in ['cross', 'c']:
        train.cross_validation()
    else:
        while True:
            cur_score = play_game()
            print('Score: {}'.format(cur_score))
            input()
