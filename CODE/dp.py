from math import sqrt

class DouglasPeuckerSimple:
    def __init__(self, epsilon):
        self.epsilon = epsilon

    def distance(self, p1, p2, p):
        x1, y1 = p1
        x2, y2 = p2
        x, y = p

        numer = abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1)
        denom = sqrt((y2 - y1)**2 + (x2 - x1)**2)
        return numer / denom if denom != 0 else 0

    def simplify(self, point_list):
        if len(point_list) < 3:
            return point_list

        dmax = 0
        index = 0
        start = point_list[0]
        end = point_list[-1]

        for i in range(1, len(point_list) - 1):
            d = self.distance(start, end, point_list[i])
            if d > dmax:
                index = i
                dmax = d

        if dmax > self.epsilon:
            left = self.simplify(point_list[:index + 1])
            right = self.simplify(point_list[index:])
            return left[:-1] + right
        else:
            return [start, end]


dp = DouglasPeuckerSimple(epsilon=1.0)
simplified = dp.simplify(original_points)

for point in simplified:
    print(point)
        
