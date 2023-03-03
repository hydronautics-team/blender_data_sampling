from math import ceil, floor
from numpy import meshgrid, linspace
# import matplotlib.pyplot as plt
import random


def make_grid(n, xlow, ylow, xhigh, yhigh, max_dim):

    grid = [[1 for _ in range(xlow + max_dim, xhigh - max_dim + 1)] for _ in range(ylow + max_dim, yhigh - max_dim + 1)]
    ans = []

    # plt.scatter(xlow, ylow, marker='x', c='r')
    # plt.scatter(xlow, yhigh, marker='x', c='r')
    # plt.scatter(xhigh,ylow, marker='x', c='red')
    # plt.scatter(xhigh, yhigh, marker='x', c='r')


    for i in range(n):

        x = random.randint(0, len(grid) - 1)
        y = random.randint(0, len(grid[0]) - 1)

        # print(x, y)

        while grid[x][y] == 0:
            x = random.randint(0, len(grid) - 1)
            y = random.randint(0, len(grid[0]) - 1)

        grid[x][y] = 0
        ans.append((x, y))

        for k in range(x - max_dim, x + max_dim + 1):
            for l in range(y - max_dim, y + max_dim + 1):
                try:
                    grid[k][l] = 0
                except IndexError:
                    pass


    # for i in range(len(grid)):
    #   print(' '.join(map(str, grid[i])))
    # for el in ans:
    #    plt.scatter(xlow + el[0], ylow + el[1])
    # plt.show()
    return ans


print(make_grid(8, -22, -55, 22, -37, 2))