import numpy as np
from random import randint, random, uniform

from . import intelligence


class hs(intelligence.sw):
    """
    Harmony Search
    """

    def __init__(self, n, function, lb, ub, dimension, iteration, par=0.5,
                 hmcr=0.5, bw=0.5, initfunc=None):
        """
        :param n: number of agents
        :param function: test function
        :param lb: lower bound for the function variables
        :param ub: upper bound for the function variables
        :param dimension: space dimension
        :param iteration: number of iterations
        :param par: pitch adjusting rate (default value is 0.5)
        :param hmcr: harmony consideration rate (default value is 0.5)
        :param bw: bandwidth (default value is 0.5)
        :param initfunc: function to initialize agents (default value is None, so that numpy.random.uniform is used)
        """

        super(hs, self).__init__()

        if not callable(initfunc):
            initfunc = np.random.uniform

        nn = n

        self.__agents = initfunc(lb, ub, (n, dimension))
        self._points(self.__agents)

        Gbest = self.__agents[np.array([function(x)
                                        for x in self.__agents]).argmin()]
        worst = np.array([function(x) for x in self.__agents]).argmax()

        for t in range(iteration):

            hnew = [0 for k in range(dimension)]

            for i in range(len(hnew)):
                if random() < hmcr:
                    hnew[i] = self.__agents[randint(0, nn - 1)][i]
                    if random() < par:
                        hnew[i] += uniform(-1, 1) * bw
                else:
                    hnew[i] = uniform(lb, ub)

            if function(hnew) < function(self.__agents[worst]):
                self.__agents[worst] = hnew
                worst = np.array([function(x) for x in self.__agents]).argmax()

            Pbest = self.__agents[
                np.array([function(x) for x in self.__agents]).argmin()]
            if function(Pbest) < function(Gbest):
                Gbest = Pbest[:]

            self._points(self.__agents)

        self._set_Gbest(Gbest)
