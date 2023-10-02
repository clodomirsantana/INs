import copy
import math
import random
import time
from operator import truediv

import numpy as np

# This code was based on in the following references:
# [1] "Discrete Binary Cat Swarm Optimization Algorithm" published by Sharafi, Khanesar and Teshnehlab.
# [2] "A Binary Cat Swarm Optimization Algorithm for the Non-Unicost Set Covering Problem" published by Crawford et al.
from scipy.spatial import distance
from networks.InteractionNetwork import InteractionNetwork


class Cat(object):
    def __init__(self, dim):
        self.pos = np.random.choice([0, 1], size=(dim,))
        self.speed = np.random.choice([0, 1], size=(dim,))
        self.cost = np.nan
        self.train_acc = 0.0
        self.test_acc = 0.0
        self.features = 0.0
        self.is_seeking = False
        self.prob = 0.0


class BCSO(object):
    def __init__(self, objective_function, swarm_size=50, n_iter=1000, w=0.4, c1=2.05, smp=3, cdc=0.2, pmo=0.2, mr=0.5,
                 output_dir=''):
        self.name = "BCSO"
        self.output_dir = output_dir
        self.objective_function = objective_function
        self.dim = objective_function.dim
        self.minf = objective_function.minf
        self.maxf = objective_function.maxf
        self.swarm_size = swarm_size
        self.n_iter = n_iter
        self.n_iter = n_iter
        self.swarm = []
        self.start_time = None

        # gbest of the swarm
        self.best_agent = Cat(self.dim)
        self.best_agent.cost = -(10 ** 100)

        self.mr = mr
        self.smp = smp
        self.cdc = cdc
        self.pmo = pmo
        self.pmo = pmo

        # Static parameters of the PSO
        self.w = w
        self.c1 = c1
        self.v_max = 50  # Based on paper [2]
        self.v_min = -50  # Based on paper [2]

        # Variables that store of state
        self.optimum_cost_tracking_iter = []
        self.swarm_cost_tracking_iter = []
        self.curr_best_cost_tracking_iter = []
        self.curr_worst_cost_tracking_iter = []
        self.execution_time_tracking_iter = []
        self.pos_diff_mean_iter = []

        self.inet = []

    def __init_swarm(self):
        self.best_agent = Cat(self.dim)
        self.best_agent.cost = -(10 ** 100)

        for i in range(self.swarm_size):
            cat = Cat(self.dim)
            cat.cost, cat.test_acc, cat.train_acc, cat.features = self.objective_function.evaluate(cat.pos)
            cat.prob = 1.0 / self.swarm_size
            self.eval_track_update()
            if self.best_agent.cost > cat.cost:
                self.best_agent = copy.deepcopy(cat)
            self.swarm.append(cat)

        self.inet = InteractionNetwork(self.swarm_size, directed=True, output_dir=self.output_dir)

    def _init_bcso(self):
        self.best_agent = None
        self.swarm = []
        self.start_time = time.time()
        self.optimum_cost_tracking_eval = []
        self.optimum_cost_tracking_iter = []

        self.optimum_train_acc_tracking_eval = []
        self.optimum_train_acc_tracking_iter = []

        self.optimum_test_acc_tracking_eval = []
        self.optimum_test_acc_tracking_iter = []

        self.optimum_features_tracking_eval = []
        self.optimum_features_tracking_iter = []

        self.execution_time_tracking_eval = []
        self.execution_time_tracking_iter = []

    def eval_track_update(self):
        self.optimum_cost_tracking_eval.append(self.best_agent.cost)
        self.optimum_train_acc_tracking_eval.append(self.best_agent.train_acc)
        self.optimum_test_acc_tracking_eval.append(self.best_agent.test_acc)
        self.optimum_features_tracking_eval.append(self.best_agent.features)
        self.execution_time_tracking_eval.append(time.time() - self.start_time)

    def iter_track_update(self):
        self.optimum_cost_tracking_iter.append(self.best_agent.cost)
        self.optimum_train_acc_tracking_iter.append(self.best_agent.train_acc)
        self.optimum_test_acc_tracking_iter.append(self.best_agent.test_acc)
        self.optimum_features_tracking_iter.append(self.best_agent.features)
        self.execution_time_tracking_iter.append(time.time() - self.start_time)

    def update_best_cat(self):
        for cat in self.swarm:
            if cat.cost >= self.best_agent.cost:
                self.best_agent = copy.deepcopy(cat)

    @staticmethod
    def roulette_wheel(swarm):
        k = range(len(swarm))
        r = np.random.uniform()
        for i in k:
            if r > swarm[i].prob:
                return i
        else:
            return np.random.choice(k)

    def mutate(self, cat):
        selected_dim = random.sample(range(0, self.dim), int(self.cdc * self.dim))
        for d in selected_dim:
            if np.random.uniform() <= self.pmo:
                cat.pos[d] = 1 - cat.pos[d]

    def create_copies(self, cat):
        copies = []
        for n in range(self.smp - 1):
            copies.append(copy.deepcopy(cat))
        return copies

    def seeking(self):
        for cat in range(self.swarm_size):
            if self.swarm[cat].is_seeking:
                copies = self.create_copies(self.swarm[cat])
                for c in copies:
                    self.mutate(c)
                    c.cost, c.test_acc, c.train_acc, c.features = self.objective_function.evaluate(c.pos)
                    self.eval_track_update()
                self.calculate_probabilities(copies)
                selected_cat = self.roulette_wheel(copies)
                self.swarm[cat] = copy.deepcopy(copies[selected_cat])

    @staticmethod
    def calculate_probabilities(swarm_c):
        max_fit = -np.inf
        min_fit = np.inf
        for cat in swarm_c:
            if cat.cost >= max_fit:
                max_fit = cat.cost
            if cat.cost <= min_fit:
                min_fit = cat.cost

        for cat in swarm_c:
            if max_fit != min_fit:
                cat.prob = abs(truediv((max_fit - cat.cost), (max_fit - min_fit)))
            else:
                cat.prob = truediv(1, len(swarm_c))

    def tracing(self):
        for cat in self.swarm:
            if not cat.is_seeking:
                for d in range(self.dim):
                    if cat.pos[d] == 0:
                        if self.best_agent.pos[d] == 0:
                            cat.speed[d] += self.w * cat.speed[d] + random.random() * self.c1
                        else:
                            cat.speed[d] += self.w * cat.speed[d] - random.random() * self.c1
                    else:
                        if self.best_agent.pos[d] == 1:
                            cat.speed[d] += self.w * cat.speed[d] + random.random() * self.c1
                        else:
                            cat.speed[d] += self.w * cat.speed[d] - random.random() * self.c1

                    if cat.speed[d] > self.v_max:
                        cat.speed[d] = self.v_max
                    elif cat.speed[d] < self.v_min:
                        cat.speed[d] = self.v_min

                    # mohamadeen
                    sigmoid_speed = truediv(1, 1 + math.pow(math.e, -cat.speed[d]))
                    if random.random() < sigmoid_speed:
                        cat.pos[d] = self.best_agent.pos[d]

                self.inet.add_link(cat, self.best_agent, np.abs(distance.hamming(self.best_agent.pos, cat.pos)),
                                   [cat.cost, self.best_agent.cost], [cat.pos, self.best_agent.pos])

    @staticmethod
    def roulette_wheel(swarm):
        k = range(len(swarm))
        r = np.random.uniform()
        for i in k:
            if r > swarm[i].prob:
                return i
        else:
            return np.random.choice(k)

    def random_choice_mode(self):
        choice = np.arange(0, self.swarm_size)
        random.shuffle(choice)
        for cat in self.swarm:
            cat.is_seeking = False
        for p in range(int(self.mr * self.swarm_size)):
            self.swarm[choice[p]].is_seeking = True

    def optimize(self):
        self._init_bcso()
        self.__init_swarm()

        for i in range(self.n_iter):
            self.random_choice_mode()
            self.seeking()
            self.tracing()
            self.update_best_cat()
            self.iter_track_update()
            # print('LOG: Iter: {} - Cost: {} '.format(i, self.best_agent.cost))
            self.inet.new_iteration(i, self.best_agent)
        self.inet.save_graphs()
