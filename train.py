from random import random, uniform
from os import system
import pandas as pd
import numpy as np
from tensorflow import keras
from config import number_input_layer, number_middle_layer, number_output_layer, optimizer, loss_function, activation
from config import n_runs, n_models, n_best, n_trains
from main import play_nogui, play_game
from crossover import new_crossover, mival, maval, create_models

sqrt2 = 1.414213
columns = ['epoch', 'model number', 'lifetime', 'score', 'rating', 'model']
columns_stats = ['epoch', 'model number', 'lifetime', 'score', 'rating']

timer = 0
cross_size = 0



# n_mutate_light = 3
# n_mutate_hard = 3

n_existing_models = 0

opts = ['Adam', 'Nadam', 'SGD', ]
acts = ['elu', 'selu', 'linear', ]
losses = ['mean_squared_error', 'mean_squared_logarithmic_error', 'cosine_proximity', ]

opts = [opts[0]]
acts = [acts[2]]
losses = [losses[1]]

df_stats = pd.DataFrame(columns=columns_stats)

X = [0]*24


def count_X(field_size, field, snake, food):
    global X
    head = snake.body[-1]

    ind = 0
    # up
    X[ind] = -1/head[0]
    ind += 1
    for i in range(1, field_size[0]+2):
        field_cell_value = field[head[0]-i][head[1]]
        if field_cell_value == 1:
            X[ind] = -1/i
            break
        if field_cell_value >= 4:
            X[ind] = 0
            break
    ind += 1
    X[ind] = int(head[0] > food[0] and head[1] == food[1])
    ind += 1

    # up-right
    X[ind] = -1/(sqrt2 * min(head[0], (field_size[1] + 1) - head[1]))
    ind += 1
    for i in range(1, min(field_size)+2):
        field_cell_value = field[head[0] - i][head[1] + i]
        if field_cell_value == 1:
            X[ind] = -1/sqrt2*i
            break
        if field_cell_value >= 4:
            X[ind] = 0
            break
    ind += 1
    X[ind] = int(head[0] > food[0] and head[1] < food[1])
    ind += 1

    # right
    X[ind] = -1 / ((field_size[1] + 1) - head[1])
    ind += 1
    for i in range(1, field_size[1]+2):
        field_cell_value = field[head[0]][head[1]+i]
        if field_cell_value == 1:
            X[ind] = -1/i
            break
        if field_cell_value >= 4:
            X[ind] = 0
            break
    ind += 1
    X[ind] = int(head[0] == food[0] and head[1] < food[1])
    ind += 1

    # down-right
    X[ind] = -1 / (sqrt2 * min((field_size[0] + 1) - head[0], (field_size[1] + 1) - head[1]))
    ind += 1
    for i in range(1, min(field_size)+2):
        field_cell_value = field[head[0] + i][head[1] + i]
        if field_cell_value == 1:
            X[ind] = -1/sqrt2*i
            break
        if field_cell_value >= 4:
            X[ind] = 0
            break
    ind += 1
    X[ind] = int(head[0] < food[0] and head[1] < food[1])
    ind += 1

    # down
    X[ind] = -1 / ((field_size[0] + 1) - head[0])
    ind += 1
    for i in range(1, field_size[0] + 2):
        field_cell_value = field[head[0]+i][head[1]]
        if field_cell_value == 1:
            X[ind] = -1/i
            break
        if field_cell_value >= 4:
            X[ind] = 0
            break
    ind += 1
    X[ind] = int(head[0] < food[0] and head[1] == food[1])
    ind += 1

    # down-left
    X[ind] = -1 / (sqrt2 * min((field_size[0] + 1) - head[0], head[1]))
    ind += 1
    for i in range(1, min(field_size) + 2):
        field_cell_value = field[head[0] + i][head[1] - i]
        if field_cell_value == 1:
            X[ind] = -1/sqrt2 * i
            break
        if field_cell_value >= 4:
            X[ind] = 0
            break
    ind += 1
    X[ind] = int(head[0] < food[0] and head[1] > food[1])
    ind += 1

    # left
    X[ind] = -1 / head[1]
    ind += 1
    for i in range(1, field_size[0] + 2):
        field_cell_value = field[head[0]][head[1] - i]
        if field_cell_value == 1:
            X[ind] = -1/i
            break
        if field_cell_value >= 4:
            X[ind] = 0
            break
    ind += 1
    X[ind] = int(head[0] == food[0] and head[1] > food[1])
    ind += 1

    # up-left
    X[ind] = -1 / (sqrt2 * min(head))
    ind += 1
    for i in range(1, min(field_size) + 2):
        field_cell_value = field[head[0] - i][head[1] - i]
        if field_cell_value == 1:
            X[ind] = -1/sqrt2 * i
            break
        if field_cell_value >= 4:
            X[ind] = 0
            break
    ind += 1
    X[ind] = int(head[0] > food[0] and head[1] > food[1])

    # X = np.array([[-i for i in X]])
    return np.array([X])


class MyModel(keras.Sequential):
    model_id = 0


def create_models(n, opt, act, loss):
    global n_existing_models
    models = []
    for i in range(n):
        init = keras.initializers.RandomUniform(minval=mival, maxval=maval)
        model = MyModel()
        n_existing_models += 1
        model.model_id = n_existing_models
        model.add(keras.layers.Dense(number_middle_layer, activation=act,
                                     kernel_initializer=init,
                                     input_dim=number_input_layer))
        model.add(keras.layers.Dense(number_middle_layer, activation=act,
                                     kernel_initializer=init))
        model.add(keras.layers.Dense(number_output_layer, activation='softmax',
                                     kernel_initializer=init))
        model.compile(optimizer=opt, loss=loss)

        models.append(model)

    return models


def run(model):
    lifetime = []
    scores = []
    for i in range(n_runs):
        ilifetime, iscore = play_nogui(model)
        lifetime.append(ilifetime)
        scores.append(iscore)
        print('.', end='')
    # average values from n_runs
    print()
    return sum(lifetime)/n_runs, 1+sum(scores)/n_runs


def train_models(n, opt=optimizer, act=activation, loss=loss_function, load_models=False):
    global df_stats
    global n_existing_models
    if load_models:
        models = create_models(n_best, opt, act, loss)
        for i, model in enumerate(models):
            model.load_weights('models\\last_epoch\\model-{}'.format(i))
    else:
        models = create_models(n_models, opt, act, loss)
    for i in range(n):
        system('cls')
        print('opt={}, act={}, loss={}'.format(opt, act, loss))
        print('Epoch {}/{}'.format(i, n-1))
        df_epoch = pd.DataFrame(columns=columns)
        for model in models:
            print('Running model №{}'.format(model.model_id), end='')
            lifetime, score = run(model)
            try:
                # rating = 10 ** score
                rating = 10**score + lifetime**score
                # rating = 10**score * (1 - 6 / lifetime)
            except ZeroDivisionError:
                rating = -lifetime
            df_model = pd.DataFrame([[i, model.model_id, lifetime, score, rating, model]], columns=columns)
            df_epoch = df_epoch.append(df_model, sort=False)
        # take n_best values and models
        df_epoch.sort_values(by='rating', inplace=True, ascending=False)
        best_models = list(df_epoch['model'][:n_best].values)
        save_models(best_models, i)
        the_best_model = best_models[0]
        play_game(the_best_model, i)
        print(df_epoch.drop('model', axis=1))
        del models
        print('Started crossing over')
        # new_models = crossing_over(best_models, opt, act, loss)
        new_models = new_crossover(best_models, opt, act, loss)

        print('Ended crossing over')
        models = new_models
        df_stats = df_stats.append(df_epoch.drop('model', axis=1), sort=False)
    del df_epoch
    del models
    del best_models
    del new_models

    return df_stats


def save_models(models, epoch):
    for i, model in enumerate(models):
        model.save_weights('models\\last_epoch\\model-{}'.format(epoch, i))


def cross_validation():
    global timer
    global cross_size
    global df_stats
    global n_existing_models
    timer = 0
    cross_size = len(opts)*len(acts)*len(losses)
    for opt in opts:
        for act in acts:
            for loss in losses:
                df_stats = pd.DataFrame(columns=columns_stats)
                n_existing_models = 0
                timer += 1
                cur_stats = train_models(n_trains, opt, act, loss)
                filename = 'cross validation results\\{}--{}--{}.csv'.format(opt, act, loss)
                cur_stats.to_csv(filename)
                s = ''
                with open(filename, 'r') as f:
                    s += f.readline()[1:]
                    for line in f:
                        s += line[2:]
                s = s.replace(',', ';')
                s = s.replace('.', ',')
                with open(filename, 'w') as f:
                    f.write(s)
