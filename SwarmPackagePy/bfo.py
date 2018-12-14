import numpy as np
from random import random

from . import intelligence


class bfo(intelligence.sw):
    """
    Bacteria Foraging Optimization
    """

    def __init__(self, n, function, lb, ub, dimension, iteration, Nc=2, Ns=12, C=0.2, Ped=1.15, initfunc=None):
        """
        :param n: number of agents
        :param function: test function
        :param lb: lower bound for the function variables
        :param ub: upper bound for the function variables
        :param dimension: space dimension
        :param iteration: the number of iterations
        :param Nc: number of chemotactic steps (default value is 2)
        :param Ns: swimming length (default value is 12)
        :param C: the size of step taken in the random direction specified by
        the tumble (default value is 0.2)
        :param Ped: elimination-dispersal probability (default value is 1.15)
        :param initfunc: function to initialize agents (default value is None, so that numpy.random.uniform is used)
        """

        super(bfo, self).__init__()
        ub_arr = np.ones((dimension, )) * ub
        lb_arr = np.ones((dimension, )) * lb

        if not callable(initfunc):
            initfunc = np.random.uniform
        self.__agents = initfunc(lb, ub, (n, dimension))
        self._points(self.__agents)
        n_is_even = True
        if n & 1:
            n_is_even = False
        J = np.array([function(x) for x in self.__agents])
        Pbest = self.__agents[J.argmin()]
        Gbest = Pbest[:]
        C_list = [C - C * 0.9 * i / iteration for i in range(iteration)]
        Ped_list = [Ped - Ped * 0.5 * i / iteration for i in range(iteration)]
        J_last = J[::1]
        for t in range(iteration):
            J_chem = [J[::1]]
            for j in range(Nc):
                for i in range(n):
                    dell = np.array(initfunc(-1, 1, dimension))
                    self.__agents[i] += C_list[t] * np.linalg.norm(dell) * dell
                    for m in range(Ns):
                        if function(self.__agents[i]) < J_last[i]:
                            J_last[i] = J[i]
                            self.__agents[i] += C_list[t] * np.linalg.norm(dell) * dell
                        else:
                            dell = np.array(initfunc(-1, 1, dimension))
                            self.__agents[i] += C_list[t] * np.linalg.norm(dell) * dell
                    self.__agents[i] = np.amax((np.vstack((self.__agents[i], lb_arr))), axis=0)
                    self.__agents[i] = np.amin((np.vstack((self.__agents[i], ub_arr))), axis=0)
                J = np.array([function(x) for x in self.__agents])
                J_chem += [J]
            J_chem = np.array(J_chem)
            J_health = [(sum(J_chem[:, i]), i) for i in range(n)]
            J_health.sort()
            alived_agents = []
            for i in J_health:
                alived_agents += [list(self.__agents[i[1]])]

            if n_is_even:
                alived_agents = 2*alived_agents[:n//2]
                self.__agents = np.array(alived_agents)
            else:
                alived_agents = 2*alived_agents[:n//2] +\
                                [alived_agents[n//2]]
                self.__agents = np.array(alived_agents)
            if t < iteration - 2:
                for i in range(n):
                    r = random()
                    if r >= Ped_list[t]:
                        self.__agents[i] = initfunc(lb, ub, dimension)
            J = np.array([function(x) for x in self.__agents])
            self._points(self.__agents)
            Pbest = self.__agents[J.argmin()]
            if function(Pbest) < function(Gbest):
                Gbest = Pbest[:]
        self._set_Gbest(Gbest)
