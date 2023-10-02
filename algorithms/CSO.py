import copy
import random
import time
from operator import truediv

import numpy as np

from networks.InteractionNetwork import InteractionNetwork


# This code was based on in the following references:
# [1] "Cat Swarm Optimization Algorithm" published by Chu, Tsai and Pan (2006).
# [2] "COMPUTATIONAL INTELLIGENCE BASED ON THE BEHAVIOR OF CATS" published by Chu, Tsai and Pan (2007).

class Cat(object):
    def __init__(self, pos):
        nan = float('nan')

        self.prob = nan
        self.cost = np.inf
        self.is_seeking = False
        self.pos = copy.deepcopy(pos)
        self.vel = [nan for _ in pos]


class CSO(object):
    def __init__(self, objective_function, search_space_initializer, swarm_size=50, n_iter=1000, c1=2.0, srd=0.2,
                 cdc=0.8, mr=0.9, smp=5, up_w=0.9, lw_w=0.4, max_vel=100, output_dir=''):
        self.name = "CSO"
        self.output_dir = output_dir
        self.search_space_initializer = search_space_initializer
        self.objective_function = objective_function
        self.dim = objective_function.dim
        self.min = objective_function.min
        self.max = objective_function.max
        self.swarm_size = swarm_size
        self.start_time = None
        self.n_iter = n_iter
        self.swarm = []

        # best of the swarm
        self.best_agent = Cat(np.zeros(self.dim))
        self.best_agent.cost = np.inf
        self.curr_best_agent = None

        self.mr = mr
        self.smp = smp
        self.cdc = cdc
        self.srd = srd  # SRD
        self.w = up_w
        self.up_w = up_w
        self.lw_w = lw_w

        # Static parameters of the PSO
        self.c1 = c1

        self.max_vel = max_vel
        self.min_vel = -self.max_vel

        # Variables that store important stats
        self.optimum_cost_tracking_iter = []
        self.swarm_cost_tracking_iter = []
        self.curr_best_cost_tracking_iter = []
        self.curr_worst_cost_tracking_iter = []
        self.execution_time_tracking_iter = []
        self.pos_diff_mean_iter = []

        self.inet = None

    def __init_swarm(self):
        self.best_agent = Cat(np.zeros(self.dim))
        self.best_agent.cost = np.inf

        positions = self.search_space_initializer.sample(self.objective_function, self.swarm_size)
        for i in range(self.swarm_size):
            cat = Cat(positions[i])
            cat.cost = self.objective_function.evaluate(cat.pos)
            cat.vel = np.random.uniform(self.min_vel, self.max_vel, self.dim)
            cat.prob = 1.0 / self.swarm_size
            if self.best_agent.cost > cat.cost:
                self.best_agent = copy.deepcopy(cat)
            self.swarm.append(cat)

        self.curr_best_agent = self.swarm[0]

    def _init_cso(self):
        self.best_agent = None
        self.curr_best_agent = None
        self.swarm = []
        self.start_time = time.time()

        self.optimum_cost_tracking_iter = []
        self.swarm_cost_tracking_iter = []
        self.curr_best_cost_tracking_iter = []
        self.curr_worst_cost_tracking_iter = []
        self.execution_time_tracking_iter = []
        self.pos_diff_mean_iter = []

        self.inet = InteractionNetwork(self.swarm_size, directed=True, output_dir=self.output_dir)

    def iter_track_update(self):
        self.optimum_cost_tracking_iter.append(self.best_agent.cost)
        self.swarm_cost_tracking_iter.append(np.mean([p.cost for p in self.swarm]))
        self.curr_best_cost_tracking_iter.append(np.min([p.cost for p in self.swarm]))
        self.curr_worst_cost_tracking_iter.append(np.max([p.cost for p in self.swarm]))
        pos_diff = [np.abs(np.linalg.norm(p.pos - self.curr_best_agent.pos)) for p in self.swarm]
        self.pos_diff_mean_iter.append(np.mean(pos_diff))
        self.execution_time_tracking_iter.append(time.time() - self.start_time)


    @staticmethod
    def roulette_wheel(swarm):
        k = list(range(len(swarm)))
        random.shuffle(k)
        r = np.random.uniform()
        for i in k:
            if r < swarm[i].prob:
                return i
        else:
            return np.random.choice(k)

    def mutate(self, cat):
        selected_dim = random.sample(range(0, self.dim), int(self.cdc * self.dim))
        for d in selected_dim:
            if np.random.random() < 0.5:
                cat.pos[d] = cat.pos[d] * (1 + self.srd)
            else:
                cat.pos[d] = cat.pos[d] * (1 - self.srd)

            if cat.pos[d] > self.max:
                cat.pos[d] = self.max
            if cat.pos[d] < self.min:
                cat.pos[d] = self.min

            cat.cost = self.objective_function.evaluate(cat.pos)

    def create_copies(self, cat):
        copies = []
        for n in range(self.smp):
            copies.append(copy.deepcopy(cat))
        return copies

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

    def seeking(self, cat):
        copies = self.create_copies(cat)
        for c in range(self.smp - 1):
            self.mutate(copies[c])
        self.calculate_probabilities(copies)
        selected_cat = self.roulette_wheel(copies)
        return copies[selected_cat].pos, copies[selected_cat].vel, copies[selected_cat].cost

    def tracing(self, cat):
        r1 = np.random.random(self.dim)
        cat.vel = self.w * cat.vel + r1 * self.c1 * (self.curr_best_agent.pos - cat.pos)
        cat.vel = np.minimum(cat.vel, self.max_vel)
        cat.vel = np.maximum(cat.vel, self.min_vel)

        cat.pos += cat.vel
        cat.pos = np.minimum(cat.pos, self.max)
        cat.pos = np.maximum(cat.pos, self.min)

        cat.cost = self.objective_function.evaluate(cat.pos)

        self.inet.add_link(cat, self.curr_best_agent,
                           np.abs(np.linalg.norm(np.array(self.curr_best_agent.pos) - cat.pos)),
                           [cat.cost, self.curr_best_agent.cost], [cat.pos, self.curr_best_agent.pos])
        return cat.pos, cat.vel, cat.cost

    def random_choice_mode(self):
        choice = np.arange(0, self.swarm_size)
        random.shuffle(choice)
        for cat in self.swarm:
            cat.is_seeking = False
        for p in range(int(self.mr * self.swarm_size)):
            self.swarm[choice[p]].is_seeking = True

    def update_best_agent(self):
        self.curr_best_agent = self.swarm[0]
        for cat in self.swarm:
            if cat.cost < self.curr_best_agent.cost:
                self.curr_best_agent = cat

        if self.curr_best_agent.cost < self.best_agent.cost:
            self.best_agent = copy.deepcopy(self.curr_best_agent)

    def update_inertia(self, i):
        self.w = self.up_w - (float(i) / self.n_iter) * (self.up_w - self.lw_w)

    def optimize(self):
        self._init_cso()
        self.__init_swarm()

        for i in range(self.n_iter):
            self.random_choice_mode()
            for cat in self.swarm:
                if not cat.is_seeking:
                    cat.pos, cat.vel, cat.cost = self.seeking(cat)
                else:
                    cat.pos, cat.vel, cat.cost = self.tracing(cat)
            self.update_inertia(i)
            self.update_best_agent()
            self.iter_track_update()

            self.inet.new_iteration(i, self.best_agent)
            # print('Func: {} Iter: {} - Cost: {}'.format(self.objective_function.name, i, self.best_agent.cost))

        self.inet.save_graphs()
