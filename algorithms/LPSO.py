import copy
import time

import numpy as np

from networks.InteractionNetwork import InteractionNetwork


# This code was based on in the following references:
# [1] "Defining a Standard for Particle Swarm Optimization" published in 2007 by Bratton and Kennedy
# [2] "Comparing Inertia Weights and Constriction Factors in Particle Swarm Optimization"


# The particle is initialized in a invalid state


class Particle(object):
    def __init__(self, pos):
        nan = float('nan')
        self.pos = copy.deepcopy(pos)
        self.vel = np.zeros(len(pos))
        self.cost = np.nan
        self.pbest_pos = self.pos
        self.pbest_cost = self.cost
        self.lbest = np.nan


class LPSO(object):
    def __init__(self, objective_function, search_space_initializer, swarm_size=50, n_iter=1000, lb_w=0.4,
                 up_w=0.9, c1=2.05, c2=2.05, v_max=100000, output_dir=''):
        self.name = "LPSO"
        self.output_dir = output_dir
        self.objective_function = objective_function
        self.search_space_initializer = search_space_initializer
        self.optimum_cost_tracking_eval = []

        self.dim = objective_function.dim
        self.min = objective_function.min
        self.max = objective_function.max
        self.swarm_size = swarm_size
        self.n_iter = n_iter
        self.start_time = None

        self.swarm = []
        self.best_agent = Particle(np.zeros(self.dim))
        self.best_agent.cost = np.inf
        self.curr_best_agent = None

        self.up_w = up_w
        self.lb_w = lb_w
        self.w = up_w
        self.c1 = c1
        self.c2 = c2
        self.v_max = min(v_max, 100000)  # Based on paper [2]

        self.optimum_cost_tracking_eval = []
        self.optimum_cost_tracking_iter = []
        self.execution_time_tracking_eval = []
        self.execution_time_tracking_iter = []

        self.inet = []

    def __init_swarm(self):
        self.inet = InteractionNetwork(self.swarm_size, directed=True, output_dir=self.output_dir)
        self.best_agent = Particle(np.zeros(self.dim))
        self.best_agent.cost = np.inf

        positions = self.search_space_initializer.sample(self.objective_function, self.swarm_size)
        for i in range(self.swarm_size):
            p = Particle(positions[i])
            p.cost = self.objective_function.evaluate(p.pos)
            p.pbest_pos = copy.deepcopy(p.pos)
            p.pbest_cost = p.cost
            p.lbest = i

            if self.best_agent.cost > p.cost:
                self.best_agent = copy.deepcopy(p)
            self.swarm.append(p)

        self.curr_best_agent = self.swarm[0]
        self.update_best_agent()

    def _init_pso(self):
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

    def iter_track_update(self):
        self.optimum_cost_tracking_iter.append(self.best_agent.cost)
        self.swarm_cost_tracking_iter.append(np.mean([p.cost for p in self.swarm]))
        self.curr_best_cost_tracking_iter.append(np.min([p.cost for p in self.swarm]))
        self.curr_worst_cost_tracking_iter.append(np.max([p.cost for p in self.swarm]))
        pos_diff = [np.abs(np.linalg.norm(p.pos - self.curr_best_agent.pos)) for p in self.swarm]
        self.pos_diff_mean_iter.append(np.mean(pos_diff))
        self.execution_time_tracking_iter.append(time.time() - self.start_time)

    def update_best_agent(self):
        self.curr_best_agent = self.swarm[0]

        for p in range(self.swarm_size):
            if self.swarm[p].pbest_cost < self.best_agent.pbest_cost:
                self.best_agent = copy.deepcopy(self.swarm[p])

        for p in range(self.swarm_size):
            for p1 in [(p - 1), (p + 1)]:
                if self.swarm[p1 % self.swarm_size].pbest_cost < self.swarm[self.swarm[p].lbest].pbest_cost:
                    self.swarm[p].lbest = p1 % self.swarm_size

    def update_inertia(self, i):
        self.w = self.up_w - (float(i) / self.n_iter) * (self.up_w - self.lb_w)

    def optimize(self):
        self._init_pso()
        self.__init_swarm()

        for i in range(self.n_iter):
            for idx in range(self.swarm_size):
                p = self.swarm[idx]
                r1 = np.random.random(self.dim)
                r2 = np.random.random(self.dim)
                local_best = self.swarm[p.lbest]
                p.vel = self.w * np.array(p.vel) + self.c1 * r1 * (p.pbest_pos - p.pos) + self.c1 * r2 * (
                        local_best.pos - p.pos)

                p.vel = np.minimum(p.vel, self.v_max)
                p.vel = np.maximum(p.vel, -self.v_max)

                p.pos = p.pos + p.vel
                p.pos = np.minimum(p.pos, self.max)
                p.pos = np.maximum(p.pos, self.min)

                p.cost = self.objective_function.evaluate(p.pos)

                if p.cost <= p.pbest_cost:
                    p.pbest_pos = p.pos
                    p.pbest_cost = p.cost

                self.inet.add_link(p, local_best, np.abs(np.linalg.norm(local_best.pos - p.pos)),
                                   [p.cost, local_best.cost], [p.pos, local_best.pos])

            self.update_inertia(i)
            self.update_best_agent()
            self.iter_track_update()
            # print('Func: {} Iter: {} - Cost: {}'.format(self.objective_function.name, i, self.best_agent.pbest_cost))
            self.inet.new_iteration(i, self.best_agent)
