# snake class

direction_coords = {'up': (-1, 0),
                    'down': (1, 0),
                    'left': (0, -1),
                    'right': (0, 1),
                    }


class Snake:
    # the first one -- tail, the last one -- head
    body = []

    def move(self, direction, is_food=False):
        if is_food:
            self.body.pop(0)
        self.body.append(direction_coords[direction])

    def create(self, field_size):
        # creating tail
        self.body.append((field_size[0] // 2), field_size[1] // 2)
        # creating head
        self.body.append((field_size[0] // 2 + 1), field_size[1] // 2 + 1)

    def hit_wall(self, field_size):
        x = self.body[-1][0]
        y = self.body[-1][1]
        if (x == 0 or x == field_size or
                y == 0 or y == field_size):
            return True
        return False

    def hit_itself(self):
        x = self.body[-1][0]
        y = self.body[-1][1]
        for i in self.body[:-1]:
            if i[0] == x or i[1] == y:
                return True
        return False
