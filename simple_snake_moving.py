from random import randint, choice


def generate_stupid_direction(field_size, head_coords):
    variants = ['up', 'right', 'down', 'left']
    v = ''
    x = head_coords[0]
    y = head_coords[1]
    if x == 1 and y == 1:
        v = variants[2]

    if x == 1 and y == field_size[1]:
        v = variants[3]

    if x == field_size[0] and y == 1:
        v = variants[1]

    if x == field_size[0] and y == field_size[1]:
        v = variants[0]

    if x == field_size[0]:
        v = variants[1]

    if x == 1:
        v = variants[3]

    if y == field_size[1]:
        v = variants[0]

    if y == 1:
        v = variants[2]
    if v == '':
        return choice(variants)
    if randint(1, 4) == 1:
        return choice(variants)
    else:
        return v


def generate_move_on_food(food_coords, head_coords):
    f_x, f_y = food_coords[0], food_coords[1]
    x, y = head_coords[0], head_coords[1]
    if x > f_x:
        return 'up'
    if x < f_x:
        return 'down'
    if y > f_y:
        return 'left'
    if y < f_y:
        return 'right'
    return 'down'
