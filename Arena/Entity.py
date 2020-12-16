from Arena.AbsDecisionMaker import AbsDecisionMaker
from Arena.constants import np, SIZE_X, SIZE_Y, DSM, AgentAction


class Entity:

    def __init__(self, decision_maker: AbsDecisionMaker = None):

        self._decision_maker = decision_maker
        self.x = -1
        self.y = -1

    def _choose_random_position(self):

        is_obs = True
        while is_obs:
            self.x = np.random.randint(0, SIZE_X)
            self.y = np.random.randint(0, SIZE_Y)
            is_obs = self.is_obs(self.x, self.y)

    def __str__(self):
        # for debugging purposes
        return f"{self.x}, {self.y}"

    def __sub__(self, other):
        return (self.x - other.x, self.y - other.y)

    def __add__(self, other):
        return (self.x + other.x, self.y + other.y)

    def get_coordinates(self):
        return self.x, self.y

    def set_coordinatess(self, x, y):
        self.x = x
        self.y = y

    def move(self, y=None, x=None):
        # if no value for x- move randomly
        if x==None:
            new_x = self.x + np.random.randint(-1, 2)
        else:
            new_x = self.x + x

        # if no value for y- move randomly
        if y==None:
            new_y = self.y + np.random.randint(-1, 2)
        else:
            new_y = self.y + y

        is_legal, new_x, new_y = self.check_if_move_is_legal(new_x, new_y)
        if is_legal:
            self.x = new_x
            self.y = new_y
        # else: stay at last spot

    def check_if_move_is_legal(self, x, y):
        # if we are out of bounds- fix!
        if x < 0:
            x = 0
        elif x > SIZE_X - 1:
            x = SIZE_X - 1
        if y < 0:
            y = 0
        elif y > SIZE_Y - 1:
            y = SIZE_Y - 1

        is_obs = self.is_obs(x, y)

        return not(is_obs), x, y

    def is_obs(self, x=-1, y=-1):
        # if first draw of start point
        if x == -1 and y == -1:
            return True

        if DSM[x][y] == 1:
            return True
        return False

    def action(self, a: AgentAction):


        """9 possible moves!"""
        if a == AgentAction.TopRight:  # (down-left)
            self.move(x=1, y=-1)
        elif a == AgentAction.Right:  # (down)
            self.move(x=1, y=0)
        elif a == AgentAction.BottomRight:  # (down-right)
            self.move(x=1, y=1)
        elif a == AgentAction.Bottom:  # (left)
            self.move(x=0, y=-1)
        elif a == AgentAction.Stay:  # stay in place!
            self.move(x=0, y=0)
        elif a == AgentAction.Top:  # (right)
            self.move(x=0, y=1)
        elif a == AgentAction.BottomLeft:  # (up-left)
            self.move(x=-1, y=-1)
        elif a==AgentAction.Left:  # (up)
            self.move(x=-1, y=0)
        elif a == AgentAction.TopLeft:  # (up-right)
            self.move(x=-1, y=1)

        # if a == AgentAction.TopRight:
        #     self.move(x=1, y=-1)
        # elif a == AgentAction.Right:
        #     self.move(x=1, y=0)
        # elif a == AgentAction.BottomRight:
        #     self.move(x=1, y=1)
        # elif a == AgentAction.Bottom:
        #     self.move(x=0, y=-1)
        # elif a == AgentAction.Stay:  # stay in place!
        #     self.move(x=0, y=0)
        # elif a == AgentAction.Top:
        #     self.move(x=0, y=1)
        # elif a == AgentAction.BottomLeft:
        #     self.move(x=-1, y=-1)
        # elif a==AgentAction.Left:
        #     self.move(x=-1, y=0)
        # elif a == AgentAction.TopLeft:
        #     self.move(x=-1, y=1)
