import random
from logging import getLogger

import numpy as np

from . import intelligence


class fwa(intelligence.sw):
    """
    Fireworks Algorithm
    """
    def __init__(self, n, function, lb, ub, dimension, iteration, m1=7, m2=7, eps=0.001, amp=2, a=0.3, b=3,
                 initfunc=None):

        """
        :param n: number of fireworks
        :param function: test function
        :param lb: lower bound for the function variables
        :param ub: upper bound for the function variables
        :param dimension: space dimension
        :param iteration: the number of iterations
        :param m1: parameter controlling the number of normal sparks (default value is 7)
        :param m2: parameter controlling the number of Gaussian sparks (default value is 7)
        :param eps: constant used to avoid division by zero (default value is 0.001)
        :param amp: amplitude of normal explosion (default value is 2)
        :param a: parameter controlling the lower bound for number of normal sparks (default value is 0.3)
        :param b: parameter controlling the upper bound for number of normal sparks, b must be greater than a (b is set to 3 by default)
        :param initfunc: function to initialize agents (default value is None, so that numpy.random.uniform is used)
        """

        super(fwa, self).__init__()

        if not callable(initfunc):
            initfunc = np.random.uniform

        self.__agents = initfunc(lb, ub, (n, dimension))
        self._points(self.__agents)

        p_best = self.__agents[np.array([function(x) for x in self.__agents]).argmin()]
        g_best = p_best[:]

        for i in range(iteration):
            getLogger().info(f"FWA Iteration {i}")
            y_max = max([function(x) for x in self.__agents])
            sparks = []
            for fw in self.__agents:
                self.__explosion_operator(sparks, fw, function, dimension, m1, eps, amp, y_max, a, b)
                self.__gaussian_mutation(sparks, fw, dimension, m2)

            self.__mapping_rule(sparks, lb, ub, dimension)
            self.__selection(sparks, n, function)
            self._points(self.__agents)

            p_best = self.__agents[
                np.array([function(x) for x in self.__agents]).argmin()]
            if function(p_best) < function(g_best):
                g_best = p_best[:]
            getLogger().info(f"Current best costs: {function(g_best):.3f}")

        self._set_Gbest(g_best)

    def __explosion_operator(self, sparks, fw, function, dimension, m, eps, amp, y_max, a, b):
        sparks_num = self.__round(m * (y_max - function(fw) + eps / (sum([y_max - function(fwk) for fwk in self.__agents]) + eps)), m, a, b)
        amplitude = amp * (function(fw) - y_max + eps) / (sum([function(fwk) - y_max for fwk in self.__agents]) + eps)
        for j in range(int(sparks_num)):
            sparks.append(np.array(fw))
            for k in range(dimension):
                if random.choice([True, False]):
                    sparks[-1][k] += random.uniform(-amplitude, amplitude)

    @staticmethod
    def __gaussian_mutation(sparks, fw, dimension, m):
        for j in range(m):
            g = np.random.normal(1, 1)
            sparks.append(np.array(fw))
            for k in range(dimension):
                if random.choice([True, False]):
                    sparks[-1][k] *= g

    @staticmethod
    def __mapping_rule(sparks, lb, ub, dimension):
        for i in range(len(sparks)):
            for j in range(dimension):
                if sparks[i][j] > ub or sparks[i][j] < lb:
                    sparks[i][j] = lb + (sparks[i][j]-lb) % (ub-lb)

    def __selection(self, sparks, n, function):
        self.__agents = sorted(np.concatenate((self.__agents,sparks)), key=function)[:n]

    @staticmethod
    def __round(s, m, a, b):
        if s < a * m:
            return round(a*m)
        elif s > b * m:
            return round(b*m)
        else:
            return round(s)

