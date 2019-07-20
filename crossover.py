from random import random, uniform
from tensorflow import keras
import numpy as np
from config import number_input_layer, number_middle_layer, number_output_layer, n_best


maval = 0.1
mival = -maval

n_existing_models = 0


class MyModel(keras.Sequential):
    model_id = 0


def create_model(act):
    global n_existing_models
    init = keras.initializers.RandomUniform(minval=mival, maxval=maval)
    model = keras.Sequential()
    n_existing_models += 1
    model.model_id = n_existing_models
    model.add(keras.layers.Dense(number_middle_layer, activation=act,
                                 kernel_initializer=init,
                                 input_dim=number_input_layer))
    model.add(keras.layers.Dense(number_middle_layer, activation=act,
                                 kernel_initializer=init))
    model.add(keras.layers.Dense(number_output_layer, activation='softmax',
                                 kernel_initializer=init))
    return model


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


def cross_genes_single(model1, model2):
    w1 = model1.get_weights()
    w2 = model2.get_weights()
    wo = []
    for i in range(len(w1)):
        w2_iter = np.nditer(w2[i])
        w = [0]*len(w2[i].flatten())
        for ind, el1 in enumerate(np.nditer(w1[i])):
            el2 = next(w2_iter)
            r_w = el1 if round(random()) == 1 else el2
            w[ind] = r_w
        wo.append(np.array(w).reshape(w1[i].shape))
    return wo


def cross_genes_bin(model1, model2):
    w1 = model1.get_weights()
    w2 = model2.get_weights()
    wo = []
    for i in range(len(w1)):
        l = len(w1[i].flat)
        p1 = list(w1[i].flat)[:l // 2]
        p2 = list(w2[i].flat)[l // 2:]
        w = np.array(p1 + p2)
        wo.append(w.reshape(w1[i].shape))
    return wo


def crossing_over_bin(models, opt, act, loss):
    new_models = []
    for i in range(n_best // 2):
        new_weights = cross_genes_bin(models[0], models[i+1])
        new_model = create_model(act)
        new_model.set_weights(new_weights)
        new_model.compile(optimizer=opt, loss=loss)
        new_models.append(new_model)

    for i in range(n_best // 2):
        new_weights = cross_genes_bin(models[i*2], models[i*2+1])
        new_model = create_model(act)
        new_model.set_weights(new_weights)
        new_model.compile(optimizer=opt, loss=loss)
        new_models.append(new_model)

    return new_models


def cross_genes_grouped(model1, model2):
    w1 = model1.get_weights()
    w2 = model2.get_weights()
    wo = []
    for i in range(len(w1)):
        if len(w1[i].shape) == 1:
            wo.append(w1[i] if round(random()) == 1 else w2[i])
        else:
            w = [0]*w1[i].shape[0]
            for ind, j in enumerate(range(w1[i].shape[0])):
                w[ind] = w1[i][j].copy() if round(random()) == 1 else w2[i][j].copy()
            wo.append(w)
    return wo


def crossing_over(models, opt, act, loss):
    new_models = []

    # for i in range(n_best // 2):
    #     new_weights = cross_genes_single(models[2*i], models[i*2+1])
    #     new_model = create_model(act)
    #     new_model.set_weights(new_weights)
    #     new_model.compile(optimizer=opt, loss=loss)
    #     new_models.append(new_model)

    for i in range(n_best // 2):
        new_weights = cross_genes_grouped(models[0], models[i + 1])
        new_model = create_model(act)
        new_model.set_weights(new_weights)
        new_model.compile(optimizer=opt, loss=loss)
        new_models.append(new_model)

    for i in range(n_best // 2):
        new_weights = cross_genes_grouped(models[2 * i], models[i * 2 + 1])
        new_model = create_model(act)
        new_model.set_weights(new_weights)
        new_model.compile(optimizer=opt, loss=loss)
        new_models.append(new_model)

    return new_models
    # new_models = []
    # for i in range(len(models)//2):
    #     for _ in range(n_models//n_best):
    #         w0, w2 = choose_genes(models[2*i], models[i*2+1])
    #         new_model = create_model()
    #         new_model.set_weights((w0, np.zeros(24), w2))
    #         new_model.compile(optimizer='Adam', loss='mean_squared_logarithmic_error')
    #         new_models.append(new_model)
    # return new_models


def new_mutate(models, opt, act, loss):
    new_models = create_models(len(models), opt, act, loss)
    c = 0
    threshold = 1 / 200
    for ind, model in enumerate(models):
        w = model.get_weights()
        for i, wi in enumerate(w):
            t = []
            for j in np.nditer(wi):
                if random() < threshold and j != 0:
                    t.append(uniform(mival, maval))
                    c += 1
                else:
                    t.append(j)
            w[i] = np.array(t).reshape(wi.shape)
        new_models[ind].set_weights(w)
        print("{} mutates for model № {}".format(c, model.model_id))
        c = 0

    return new_models


def new_cross_over(models, opt, act, loss):
    new_models = list()

    for i in range(n_best//2):
        new_weights = cross_genes_grouped(models[0], models[i + 1])
        new_model = create_model(act)
        new_model.set_weights(new_weights)
        new_model.compile(optimizer=opt, loss=loss)
        new_models.append(new_model)

    for i in range(n_best // 2):
        new_weights = cross_genes_grouped(models[2 * i], models[i * 2 + 1])
        new_model = create_model(act)
        new_model.set_weights(new_weights)
        new_model.compile(optimizer=opt, loss=loss)
        new_models.append(new_model)

    return new_models


def new_cross_over_bin(models, opt, act, loss):
    new_models = list()

    for i in range(n_best // 2):
        new_weights = cross_genes_bin(models[0], models[i+1])
        new_model = create_model(act)
        new_model.set_weights(new_weights)
        new_model.compile(optimizer=opt, loss=loss)
        new_models.append(new_model)

    return new_models


def new_cross_over_single(models, opt, act, loss):
    new_models = list()

    for i in range(n_best // 2):
        new_weights = cross_genes_single(models[0], models[i + 1])
        new_model = create_model(act)
        new_model.set_weights(new_weights)
        new_model.compile(optimizer=opt, loss=loss)
        new_models.append(new_model)

    return new_models


def new_crossover(models, opt, act, loss):
    new_models = list()
    new_models.extend(models)
    new_models.extend(new_mutate(models, opt, act, loss))
    print('mutated')
    new_models.extend(new_cross_over(models, opt, act, loss))
    print('crossed over')
    new_models.extend(new_cross_over_bin(models, opt, act, loss))
    print('crossed over bin')
    new_models.extend(new_cross_over_single(models, opt, act, loss))
    print('crossed over single')
    return new_models
