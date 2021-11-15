import math
class Point:

    epsilon = 10e-5
    def __init__(self):
        pass
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def distance(self, p:Point):
        return math.sqrt((self.x-p.x)*(self.x-p.x) + (self.y-p.y)*(self.y-p.y) )
