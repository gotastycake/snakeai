from random import random

import pandas as pd
import numpy as np
from tensorflow import keras
from config import number_input_layer, number_middle_layer, number_output_layer
from main import play_nogui, play_game

sqrt2 = 1.414213

n_models = 12
n_runs = 5
n_best = 6
n_mutate_light = 3
n_mutate_hard = 3

mival = -2
maval = 2

df_stats = pd.DataFrame(columns=['epoch', 'model number', 'steps', 'score', 'rating'])


def count_X(field_size, field, snake, food):
    head = snake.body[-1]

    X = list()

    # up
    X.append(-1/head[0])
    for i in range(1, field_size[0]+2):
        field_cell_value = field[head[0]-i][head[1]]
        if field_cell_value == 1:
            X.append(-1/i)
            break
        if field_cell_value >= 4:
            X.append(0)
            break
    X.append(int(head[0] > food[0] and head[1] == food[1]))

    # up-right
    X.append(-1/(sqrt2 * min(head[0], (field_size[1] + 1) - head[1])))
    for i in range(1, min(field_size)+2):
        field_cell_value = field[head[0] - i][head[1] + i]
        if field_cell_value == 1:
            X.append(-1/sqrt2*i)
            break
        if field_cell_value >= 4:
            X.append(0)
            break
    X.append(int(head[0] > food[0] and head[1] < food[1]))

    # right
    X.append(-1 / ((field_size[1] + 1) - head[1]))
    for i in range(1, field_size[1]+2):
        field_cell_value = field[head[0]][head[1]+i]
        if field_cell_value == 1:
            X.append(-1/i)
            break
        if field_cell_value >= 4:
            X.append(0)
            break
    X.append(int(head[0] == food[0] and head[1] < food[1]))

    # down-right
    X.append(-1 / (sqrt2 * min((field_size[0] + 1) - head[0], (field_size[1] + 1) - head[1])))
    for i in range(1, min(field_size)+2):
        field_cell_value = field[head[0] + i][head[1] + i]
        if field_cell_value == 1:
            X.append(-1/sqrt2*i)
            break
        if field_cell_value >= 4:
            X.append(0)
            break
    X.append(int(head[0] < food[0] and head[1] < food[1]))

    # down
    X.append(-1 / ((field_size[0] + 1) - head[0]))
    for i in range(1, field_size[0] + 2):
        field_cell_value = field[head[0]+i][head[1]]
        if field_cell_value == 1:
            X.append(-1/i)
            break
        if field_cell_value >= 4:
            X.append(0)
            break
    X.append(int(head[0] < food[0] and head[1] == food[1]))

    # down-left
    X.append(-1 / (sqrt2 * min((field_size[0] + 1) - head[0], head[1])))
    for i in range(1, min(field_size) + 2):
        field_cell_value = field[head[0] + i][head[1] - i]
        if field_cell_value == 1:
            X.append(-1/sqrt2 * i)
            break
        if field_cell_value >= 4:
            X.append(0)
            break
    X.append(int(head[0] < food[0] and head[1] > food[1]))

    # left
    X.append(-1 / head[1])
    for i in range(1, field_size[0] + 2):
        field_cell_value = field[head[0]][head[1] - i]
        if field_cell_value == 1:
            X.append(-1/i)
            break
        if field_cell_value >= 4:
            X.append(0)
            break
    X.append(int(head[0] == food[0] and head[1] > food[1]))

    # up-left
    X.append(-1 / (sqrt2 * min(head)))
    for i in range(1, min(field_size) + 2):
        field_cell_value = field[head[0] - i][head[1] - i]
        if field_cell_value == 1:
            X.append(-1/sqrt2 * i)
            break
        if field_cell_value >= 4:
            X.append(0)
            break
    X.append(int(head[0] > food[0] and head[1] > food[1]))

    X = np.array([X])
    return X


def create_models(n):
    models = []
    for i in range(n):
        init = keras.initializers.RandomUniform(minval=mival, maxval=maval)
        model = keras.Sequential()
        model.add(keras.layers.Dense(number_middle_layer, activation='linear',
                                     kernel_initializer=init,
                                     input_dim=number_input_layer))
        # model.add(keras.layers.Dense(number_middle_layer, activation='tanh',
        #                              kernel_initializer=init))
        model.add(keras.layers.Dense(number_output_layer, activation='softmax',
                                     kernel_initializer=init))
        model.compile(optimizer='Adam', loss='mean_squared_logarithmic_error')

        models.append(model)

    return models


def create_model():
    init = keras.initializers.RandomUniform(minval=mival, maxval=maval)
    model = keras.Sequential()
    model.add(keras.layers.Dense(number_middle_layer, activation='linear',
                                 kernel_initializer=init,
                                 input_dim=number_input_layer))
    # model.add(keras.layers.Dense(number_middle_layer, activation='tanh',
    #                             kernel_initializer=init))
    model.add(keras.layers.Dense(number_output_layer, activation='softmax',
                                 kernel_initializer=init))
    #model.compile(optimizer='SGD', loss='mean_squared_error')

    return model


def choose_genes(model1, model2):
    w1 = model1.get_weights()
    w2 = model2.get_weights()
    o_w0 = []
    for i in range(number_middle_layer):
        o_w0.append([])
        for j in range(len(w1[0])):
            r_w = w1[0][i][j] if round(random()) == 1 else w2[0][i][j]
            o_w0[-1].append(r_w)

    o_w2 = []
    for i in range(number_middle_layer):
        r_w = w1[2][i] if round(random()) == 1 else w2[2][i]
        o_w2.append(r_w)
    return np.array(o_w0), np.array(o_w2)


def crossing_over(models):
    new_models = []
    for i in range(len(models)//2):
        #print('Crossing model {} and model {}'.format(2*i, 2*i+1), end='')
        for _ in range(n_models//n_best):
            #print('.', end='')
            w0, w2 = choose_genes(models[2*i], models[i*2+1])
            new_model = create_model()
            new_model.set_weights((w0, np.zeros(24), w2))
            new_model.compile(optimizer='Adam', loss='mean_squared_logarithmic_error')
            new_models.append(new_model)
        #print()
    return new_models


def run(model):
    steps = []
    scores = []
    for i in range(n_runs):
        print('.', end='')
        isteps, iscore = play_nogui(model)
        steps.append(isteps)
        scores.append(iscore)
        #print(isteps, sep='', end='')
    # average values from n_runs
    return sum(steps)/n_runs, 1+sum(scores)/n_runs


def train_models(n):
    global df_stats
    models = create_models(n_models)
    for i in range(n):
        print('Epoch {}'.format(i))
        df_epoch = pd.DataFrame(columns=['epoch', 'model number', 'steps', 'score', 'rating'])
        for model_number, model in enumerate(models):
            print('Running model №{}'.format(model_number), end='')
            steps, score = run(model)
            print()
            try:
                rating = 10**score + steps**score
            except ZeroDivisionError:
                rating = -steps
            df_model = pd.DataFrame([[i, model_number, steps, score, rating]], columns=['epoch', 'model number', 'steps', 'score', 'rating'])
            df_epoch = df_epoch.append(df_model)
        # take n_best values and models
        df_epoch.sort_values('rating', inplace=True, ascending=False)

        best_models = [models[i] for i in df_epoch['model number'][:n_best].values]
        the_best_model = best_models[0]
        play_game(the_best_model)
        #print('Started crossing over')
        new_models = best_models + crossing_over(best_models)
        #print('Ended crossing over')
        models = new_models
        print(df_epoch)
        df_stats = df_stats.append(df_epoch)
