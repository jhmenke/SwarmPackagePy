from math import gamma, pi, sin
import numpy as np
from random import normalvariate, randint, random

from . import intelligence


class cso(intelligence.sw):
    """
    Cuckoo Search Optimization
    """

    def __init__(self, n, eval_func, lb, ub, dimension, iteration, pa=0.25, nest=100, initfunc=None):
        """
        :param n: number of agents
        :param eval_func: test function
        :param lb: lower bound for the function variables
        :param ub: upper bound for the function variables
        :param dimension: space dimension
        :param iteration: number of iterations
        :param pa: probability of cuckoo's egg detection (default value is 0.25)
        :param nest: number of nests (default value is 100)
        :param initfunc: function to initialize agents (default value is None, so that numpy.random.uniform is used)
        """

        super(cso, self).__init__()

        self.__Nests = []

        beta = 3 / 2
        sigma = (gamma(1 + beta) * sin(pi * beta / 2) / (gamma((1 + beta) / 2) * beta *
                                                         2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.array([normalvariate(0, 1) for _ in range(dimension)]) * sigma
        v = np.array([normalvariate(0, 1) for _ in range(dimension)])
        step = u / abs(v) ** (1 / beta)

        if not callable(initfunc):
            initfunc = np.random.uniform
        self.__agents = initfunc(lb, ub, (n, dimension))
        self.__nests = initfunc(lb, ub, (nest, dimension))
        Pbest = self.__nests[np.array([eval_func(x) for x in self.__nests]).argmin()]
        Gbest = Pbest[:]
        self._points(self.__agents)

        for t in range(iteration):

            for i in self.__agents:
                val = randint(0, nest - 1)
                if eval_func(i) < eval_func(self.__nests[val]):
                    self.__nests[val] = i

            fnests = [(eval_func(self.__nests[i]), i) for i in range(nest)]
            fnests.sort()
            fcuckoos = [(eval_func(self.__agents[i]), i) for i in range(n)]
            fcuckoos.sort(reverse=True)

            nworst = nest // 2
            worst_nests = [fnests[-i - 1][1] for i in range(nworst)]

            for i in worst_nests:
                if random() < pa:
                    self.__nests[i] = initfunc(lb, ub, (1, dimension))

            mworst = n if nest > n else nest

            for i in range(mworst):
                if fnests[i][0] < fcuckoos[i][0]:
                    self.__agents[fcuckoos[i][1]] = self.__nests[fnests[i][1]]

            self.__nests = np.clip(self.__nests, lb, ub)
            self.__levy_fly(step, Pbest, n, dimension)
            self.__agents = np.clip(self.__agents, lb, ub)
            self._points(self.__agents)
            self.__nest()

            Pbest = self.__nests[np.array([eval_func(x) for x in self.__nests]).argmin()]

            if eval_func(Pbest) < eval_func(Gbest):
                Gbest = Pbest[:]

        self._set_Gbest(Gbest)

    def __nest(self):
        self.__Nests.append([list(i) for i in self.__nests])

    def __levy_fly(self, step, Pbest, n, dimension):

        for i in range(n):
            stepsize = 0.5 * step * (self.__agents[i] - Pbest)
            self.__agents[i] += stepsize * np.array([normalvariate(0, 1) for _ in range(dimension)])

    def get_nests(self):
        """Return a history of cuckoos nests (return type: list)"""
        return self.__Nests
