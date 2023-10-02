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


class PSOPS(object):
    def __init__(self, objective_function, search_space_initializer, swarm_size=50, n_iter=1000, lb_w=0.4,
                 up_w=0.9, c1=2.05, c2=2.05, v_max=100000, radius=2.0, delta=1.0, output_dir=''):
        self.name = "PSOPS"
        self.output_dir = output_dir
        self.objective_function = objective_function
        self.search_space_initializer = search_space_initializer

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

        self.optimum_cost_tracking_iter = []
        self.swarm_cost_tracking_iter = []
        self.curr_best_cost_tracking_iter = []
        self.curr_worst_cost_tracking_iter = []
        self.execution_time_tracking_iter = []
        self.pos_diff_mean_iter = []

        self.radius = radius
        self.delta = delta

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
        for p in self.swarm:
            if p.pbest_cost < self.curr_best_agent.pbest_cost:
                self.curr_best_agent = p

        if self.curr_best_agent.pbest_cost < self.best_agent.pbest_cost:
            self.best_agent = copy.deepcopy(self.curr_best_agent)

    def update_inertia(self, i):
        self.w = self.up_w - (float(i) / self.n_iter) * (self.up_w - self.lb_w)

    def gen_neighbours(self, pos, delta):
        res = []
        for i in range(self.dim):
            line = []
            for j in range(self.dim):
                if i == j:
                    line.extend([pos[i] + delta, (pos[i] - delta)])
                else:
                    line.extend([pos[i], pos[i]])
            res.append(line)
        return np.array(res).T

    def pattern_search(self, p):
        curr_delta = self.delta
        curr_pos = copy.deepcopy(p.pos)
        while curr_delta > (self.delta / 8.0):
            neighbours = self.gen_neighbours(curr_pos, curr_delta)
            for n in neighbours:
                cost = self.objective_function.evaluate(n)
                if cost < p.cost:
                    p.pos = n
                    p.cost = cost
                    curr_delta = self.delta
                    if p.cost <= p.pbest_cost:
                        p.pbest_pos = p.pos
                        p.pbest_cost = p.cost
            else:
                curr_delta = curr_delta / 2

    def prob_selection(self):
        fits_idx = [[self.swarm[p].cost, p] for p in range(self.swarm_size)]
        res = []
        while len(fits_idx) > 0:
            best_ = np.argmin([f[0] for f in fits_idx])
            worst_ = np.argmax([f[0] for f in fits_idx])
            denomi_ = sum([fits_idx[worst_][0] - y[0] for y in fits_idx])
            if denomi_ != 0.0:
                prob_ = (fits_idx[worst_][0] - fits_idx[best_][0]) / denomi_
            else:
                prob_ = 1.0 / len(fits_idx)

            if prob_ >= np.random.random():
                res.append(fits_idx[best_][1])
                fits_idx.remove(fits_idx[best_])
                for cp in fits_idx:
                    a = self.swarm[res[-1]].pos
                    b = self.swarm[cp[1]].pos
                    if np.linalg.norm(a - b) < self.radius:
                        fits_idx.remove(cp)
        return res

    def optimize(self):
        self._init_pso()
        self.__init_swarm()

        for i in range(self.n_iter):
            for p in self.swarm:
                r1 = np.random.random(self.dim)
                r2 = np.random.random(self.dim)

                p.vel = self.w * np.array(p.vel) + self.c1 * r1 * (p.pbest_pos - p.pos) + self.c1 * r2 * (
                        self.curr_best_agent.pos - p.pos)

                p.vel = np.minimum(p.vel, self.v_max)
                p.vel = np.maximum(p.vel, -self.v_max)

                p.pos = p.pos + p.vel
                p.pos = np.minimum(p.pos, self.max)
                p.pos = np.maximum(p.pos, self.min)

                p.cost = self.objective_function.evaluate(p.pos)
                self.inet.add_link(p, self.curr_best_agent, np.abs(np.linalg.norm(self.curr_best_agent.pos - p.pos)),
                                   [p.cost, self.curr_best_agent.cost], [p.pos, self.curr_best_agent.pos])

                if p.cost <= p.pbest_cost:
                    p.pbest_pos = p.pos
                    p.pbest_cost = p.cost

            self.update_inertia(i)
            self.update_best_agent()

            # Memetic part
            selected_p = self.prob_selection()
            for p in selected_p:
                self.pattern_search(self.swarm[p])

            self.update_best_agent()
            self.iter_track_update()
            # print('Func: {} Iter: {} - Cost: {}'.format(self.objective_function.name, i, self.best_agent.pbest_cost))
            self.inet.new_iteration(i, self.best_agent)
