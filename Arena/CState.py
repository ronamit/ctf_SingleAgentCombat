from Arena.Position import Position


class State(object):

    def __init__(self, my_pos: Position, enemy_pos: Position):

        self.my_pos = my_pos
        self.enemy_pos = enemy_pos



