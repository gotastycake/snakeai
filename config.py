# Global configs


# numbers for field matrix describing snake, field and food numbers on the field
num_snake = 1
num_food = 2
num_field = 0

# constants
file_settings = 'settings.txt'

number_input_layer = 24
number_middle_layer = 24
number_output_layer = 4

optimizer = 'Adam'
activation = 'linear'
loss_function = 'mean_squared_logarithmic_error'

n_models = 12
n_best = 6
n_runs = 10
n_trains = 15

settings = {}  # dict of settings imported from "settings" file
delay = 0  # delay between frames
field_size = ()

directions = {
    0: 'up',
    1: 'right',
    2: 'down',
    3: 'left',
}


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

load_settings()
