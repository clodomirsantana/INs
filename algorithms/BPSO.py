from __future__ import division

import copy
import time
import numpy as np
from scipy.spatial import distance

# This code was based on in the following references:
# [1] "A discrete binary version of the particle swarm algorithm" published in 1997 by J Kennedy and RC Eberhart
from networks.InteractionNetwork import InteractionNetwork


class BinaryParticle(object):
    BINARY_BASE = 2

    def __init__(self, dim, maximize=True):
        self.dim = dim
        self.pos = BinaryParticle.__initialize_position(dim)
        self.speed = np.zeros((1, dim), dtype=np.float32).reshape(dim)
        self.cost = -(10 ** 100) if maximize else 10 ** 100
        self.train_acc = 0.0
        self.test_acc = 0.0
        self.features = 0.0
        self.pbest_pos = self.pos
        self.pbest_cost = self.cost

    def update_components(self, w, c1, c2, v_max, gbest, inet):
        r1 = np.random.random(self.dim)
        r2 = np.random.random(self.dim)
        self.speed = w * self.speed + c1 * r1 * (self.pbest_pos - self.pos) + \
                     c2 * r2 * (gbest.pos - self.pos)
        self.restrict_vmax(v_max)
        self.update_pos()
        inet.add_link(self, gbest, np.abs(distance.hamming(gbest.pos, self.pos)),
                      [self.cost, gbest.cost], [self.pos, gbest.pos])

    def restrict_vmax(self, v_max):
        self.speed[self.speed > v_max] = v_max
        self.speed[self.speed < -v_max] = -v_max

    def update_pos(self):
        probs = []
        for d in self.speed:
            probs.append(self.__sgm(d))
        prob = np.random.random(self.dim)
        self.pos[probs > prob] = 1
        self.pos[probs < prob] = 0

    @staticmethod
    def __sgm(v):
        return 1 / (1 + np.exp(-v))

    @staticmethod
    def __initialize_position(dim):
        return np.random.randint(BinaryParticle.BINARY_BASE, size=dim)


class BPSO(object):
    def __init__(self, objective_function, pop_size=1000, max_iter=5000, lb_w=0.4, up_w=0.9, c1=2.05, c2=2.05,
                 v_max=100000, maximize=True, output_dir=''):
        self.name = "BPSO"
        self.output_dir = output_dir
        self.c1 = c1
        self.c2 = c2
        self.w = up_w
        self.lb_w = lb_w
        self.up_w = up_w
        self.v_max = min(v_max, 100000)
        self.dim = objective_function.dim
        self.objective_function = objective_function
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.maximize = maximize
        self.op = max if maximize else min

        self.start_time = None

        # Variables that store of state
        self.optimum_cost_tracking_iter = []
        self.swarm_cost_tracking_iter = []
        self.curr_best_cost_tracking_iter = []
        self.curr_worst_cost_tracking_iter = []
        self.execution_time_tracking_iter = []
        self.pos_diff_mean_iter = []

        self.best_agent = None
        self.inet = []

    def __iter_track_update(self):
        self.optimum_cost_tracking_iter.append(self.best_agent.cost)
        self.swarm_cost_tracking_iter.append(np.mean([p.cost for p in self.swarm]))
        self.curr_best_cost_tracking_iter.append(np.min([p.cost for p in self.swarm]))
        self.curr_worst_cost_tracking_iter.append(np.max([p.cost for p in self.swarm]))
        pos_diff = [np.abs(np.linalg.norm(p.pos - self.best_agent.pos)) for p in self.swarm]
        self.pos_diff_mean_iter.append(np.mean(pos_diff))
        self.execution_time_tracking_iter.append(time.time() - self.start_time)

    def __init_swarm(self):
        self.w = self.up_w
        self.swarm = []
        self.best_agent = None
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

        for _ in range(self.pop_size):
            self.swarm.append(BinaryParticle(self.dim, self.maximize))

        self.inet = InteractionNetwork(self.pop_size, directed=True, output_dir=self.output_dir)

    def __evaluate_swarm(self):
        for p in self.swarm:
            self.__evaluate(self.objective_function, self.op, p)

    def __select_best_particle(self):
        current_optimal = copy.deepcopy(self.op(self.swarm, key=lambda p: p.cost))
        if not self.best_agent:
            self.best_agent = copy.deepcopy(current_optimal)
            return
        if self.best_agent.cost != self.op(self.best_agent.cost, current_optimal.cost):
            self.best_agent = copy.deepcopy(current_optimal)

    def __update_components(self):
        for p in self.swarm:
            self.__update_swarm_components(self.up_w, self.c1, self.c2, self.v_max, self.best_agent, p, self.inet)

    def __update_inertia_weight(self, itr):
        self.w = self.up_w - (float(itr) / self.max_iter) * (self.up_w - self.lb_w)

    def __evaluate(self, fitness, op, particle):
        particle.cost, particle.test_acc, particle.train_acc, particle.features = fitness.evaluate(particle.pos)
        particle.pbest_cost = op(particle.cost, particle.pbest_cost)
        if particle.pbest_cost == particle.cost:
            particle.pbest_pos = particle.pos

    @staticmethod
    def __update_swarm_components(w, c1, c2, vmax, gbest, particle, inet):
        particle.update_components(w, c1, c2, vmax, gbest, inet)

    def optimize(self):
        self.__init_swarm()
        self.__select_best_particle()

        for i in range(self.max_iter):
            self.__update_components()
            self.__evaluate_swarm()
            self.__select_best_particle()
            self.__update_inertia_weight(i)
            self.__iter_track_update()
            # print('LOG: Iter: {} - Cost: {}'.format(i, self.best_agent.cost))
            self.inet.new_iteration(i, self.best_agent)
        self.inet.save_graphs()
