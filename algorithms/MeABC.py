import copy
import time

import numpy as np
from networks.InteractionNetwork import InteractionNetwork


# This code was based on in the following references:
# [1] "Memetic search in artificial bee colony algorithm" published in 2013 by Jagdish Chand Bansal,Harish Sharma,
# K. V. Arya and Atulya Nagar

# This implementation consider #_employed = #_onlookers = #_food_sources = colony_size / 2

class Bee(object):
    def __init__(self, dim):
        nan = float('nan')
        self.pos = [nan for _ in range(dim)]
        self.cost = np.inf
        self.fitness = 0.0
        self.prob = 0.0
        self.trials = 0


class MeABC(object):
    def __init__(self, objective_function, search_space_initializer, n_iter=1000, colony_size=100, trials_limit=100,
                 c=1.5, epsilon=0.01, pr=0.4, output_dir=''):
        self.name = "MeABC"
        self.output_dir = output_dir
        self.objective_function = objective_function
        self.search_space_initializer = search_space_initializer

        self.dim = objective_function.dim
        self.minf = objective_function.min
        self.maxf = objective_function.max
        self.n_iter = n_iter

        self.start_time = None
        self.best_agent = None
        self.curr_best_agent = None

        self.num_fs = int(colony_size / 2)
        self.trials_limit = trials_limit
        self.food_sources = []

        self.optimum_cost_tracking_iter = []
        self.swarm_cost_tracking_iter = []
        self.curr_best_cost_tracking_iter = []
        self.curr_worst_cost_tracking_iter = []
        self.execution_time_tracking_iter = []
        self.pos_diff_mean_iter = []

        self.inet = []
        self.scout = []

        # Modifications for the MeABC
        self.c = c
        self.pr = pr
        self.epsilon = epsilon

    @staticmethod
    def calculate_fitness(cost):
        if cost >= 0:
            result = 1.0 / (1.0 + cost)
        else:
            result = 1.0 + abs(cost)
        return result

    def calculate_probabilities(self):
        sum_fit = 0.0
        for fs in range(self.num_fs):
            sum_fit += self.food_sources[fs].fitness

        for fs in range(self.num_fs):
            self.food_sources[fs].prob = (self.food_sources[fs].fitness / sum_fit)

    def update_best_solution(self):
        self.curr_best_agent = self.food_sources[0]
        for bee in self.food_sources:
            if bee.cost < self.curr_best_agent.cost:
                self.curr_best_agent = bee
            if bee.cost < self.best_agent.cost:
                self.best_agent.pos = bee.pos
                self.best_agent.cost = bee.cost

    def init_bee(self, pos):
        bee = Bee(self.dim)
        bee.pos = pos
        bee.cost = self.objective_function.evaluate(bee.pos)
        bee.fitness = self.calculate_fitness(bee.cost)
        return bee

    def init_colony(self):
        self.inet = InteractionNetwork(self.num_fs, directed=True, output_dir=self.output_dir)
        self.best_agent = Bee(self.dim)
        self.best_agent.cost = np.inf

        positions = self.search_space_initializer.sample(self.objective_function, self.num_fs)
        for i in range(self.num_fs):
            bee = self.init_bee(positions[i])
            self.food_sources.append(bee)

            if bee.cost < self.best_agent.cost:
                self.best_agent.pos = bee.pos
                self.best_agent.cost = bee.cost
        self.curr_best_agent = self.food_sources[0]

    def employed_bee_phase(self):
        for bee in range(self.num_fs):
            k = list(range(self.num_fs))
            k.remove(bee)
            k = np.random.choice(np.array(k))
            j = np.random.choice(range(self.dim))
            phi = np.random.uniform(-1, 1)
            psi = np.random.uniform(0, self.c)

            new_pos = np.copy(self.food_sources[bee].pos)
            new_pos[j] = self.food_sources[bee].pos[j] + \
                         phi * (self.food_sources[bee].pos[j] - self.food_sources[k].pos[j]) + \
                         psi * (self.curr_best_agent.pos[j] - self.food_sources[bee].pos[j])

            if new_pos[j] < self.minf:
                new_pos[j] = self.minf
            elif new_pos[j] > self.maxf:
                new_pos[j] = self.maxf
            cost = self.objective_function.evaluate(new_pos)
            fit = self.calculate_fitness(cost)

            if fit > self.food_sources[bee].fitness:
                self.food_sources[bee].pos = new_pos
                self.food_sources[bee].cost = cost
                self.food_sources[bee].fitness = fit
                self.food_sources[bee].trials = 0
                self.inet.add_link(bee, k, np.abs(np.linalg.norm(new_pos - self.food_sources[k].pos)),
                                   [cost, self.food_sources[k].cost],
                                   [new_pos, self.food_sources[k].pos])

                self.inet.add_link(bee, self.food_sources.index(self.curr_best_agent),
                                   np.abs(np.linalg.norm(self.curr_best_agent.pos - new_pos)),
                                   [cost, self.curr_best_agent.cost], [new_pos, self.curr_best_agent.pos])

            else:
                self.food_sources[bee].trials += 1

    def onlooker_bee_phase(self):
        t = s = 0
        while t < self.num_fs:
            s = (s + 1) % self.num_fs
            r = np.random.uniform()

            if r < self.food_sources[s].prob:
                t += 1

                k = list(range(self.num_fs))
                k.remove(s)
                k = np.random.choice(np.array(k))
                j = np.random.choice(range(self.dim))
                phi = np.random.uniform(-1, 1)
                psi = np.random.uniform(0, self.c)

                new_pos = np.copy(self.food_sources[s].pos)
                new_pos[j] = new_pos[j] + phi * (new_pos[j] - self.food_sources[k].pos[j]) + \
                             psi * (self.curr_best_agent.pos[j] - new_pos[j])

                if new_pos[j] < self.minf:
                    new_pos[j] = self.minf
                elif new_pos[j] > self.maxf:
                    new_pos[j] = self.maxf

                cost = self.objective_function.evaluate(new_pos)
                fit = self.calculate_fitness(cost)

                if fit > self.food_sources[s].fitness:
                    self.food_sources[s].pos = new_pos
                    self.food_sources[s].cost = cost
                    self.food_sources[s].fitness = fit
                    self.food_sources[s].trials = 0
                    self.inet.add_link(s, k, np.abs(np.linalg.norm(new_pos - self.food_sources[k].pos)),
                                       [cost, self.food_sources[k].cost], [new_pos, self.food_sources[k].pos])

                    self.inet.add_link(s, self.food_sources.index(self.curr_best_agent),
                                       np.abs(np.linalg.norm(self.curr_best_agent.pos - new_pos)),
                                       [cost, self.best_agent.cost], [new_pos, self.curr_best_agent.pos])
                else:
                    self.food_sources[s].trials += 1

    def gen_sol(self, phi):
        k = list(range(self.num_fs))
        k.remove(self.food_sources.index(self.curr_best_agent))
        k = np.random.choice(np.array(k))
        new_pos = np.copy(self.curr_best_agent.pos)

        v = []
        for j in range(self.dim):
            if np.random.uniform() > self.pr:
                v.append((new_pos[j] - self.food_sources[k].pos[j]))
                new_pos[j] = new_pos[j] + phi * v[-1]

                if new_pos[j] < self.minf:
                    new_pos[j] = self.minf
                elif new_pos[j] > self.maxf:
                    new_pos[j] = self.maxf

        return new_pos, k, np.abs(np.linalg.norm(new_pos - self.food_sources[k].pos))

    def memetic_search_phase(self):
        a, b, gr = -1.2, 1.2, 0.618

        while np.abs(a - b) > self.epsilon:
            f1 = b - (b - a) * gr
            f2 = a + (b - a) * gr

            x_new1, k1, v1 = self.gen_sol(f1)
            x_new2, k2, v2 = self.gen_sol(f2)

            cost1 = self.objective_function.evaluate(x_new1)
            cost2 = self.objective_function.evaluate(x_new2)

            if cost1 < cost2:
                b = f2
                if cost1 < self.curr_best_agent.cost:
                    self.inet.add_link(self.food_sources.index(self.curr_best_agent), k1, v1,
                                       [cost1, self.food_sources[k1].cost],
                                       [x_new1, self.food_sources[k1].pos])

                    self.curr_best_agent.pos = x_new1
                    self.curr_best_agent.cost = cost1
                    self.curr_best_agent.fitness = self.calculate_fitness(cost1)
            else:
                a = f1
                if cost2 < self.curr_best_agent.cost:
                    self.inet.add_link(self.food_sources.index(self.curr_best_agent), k2, v2,
                                       [cost2, self.food_sources[k2].cost],
                                       [x_new2, self.food_sources[k2].pos])
                    self.curr_best_agent.pos = x_new2
                    self.curr_best_agent.cost = cost2
                    self.curr_best_agent.fitness = self.calculate_fitness(cost2)

    def get_max_trial(self):
        max_ = 0
        for fs in range(self.num_fs):
            if self.food_sources[fs].trials > self.food_sources[max_].trials:
                max_ = fs
        return max_

    def scout_bee_phase(self):
        c = 0
        for fs in range(self.num_fs):
            if self.food_sources[fs].trials >= self.trials_limit:
                c += 1
                pos = self.search_space_initializer.sample(self.objective_function, 1)[0]
                self.food_sources[fs].pos = copy.deepcopy(pos)
                self.food_sources[fs].cost = self.objective_function.evaluate(self.food_sources[fs].pos)
                self.food_sources[fs].fitness = self.calculate_fitness(self.food_sources[fs].cost)
                self.food_sources[fs].trials = 0
        return c

    def iter_track_update(self):
        self.optimum_cost_tracking_iter.append(self.best_agent.cost)
        self.swarm_cost_tracking_iter.append(np.mean([p.cost for p in self.food_sources]))
        self.curr_best_cost_tracking_iter.append(np.min([p.cost for p in self.food_sources]))
        self.curr_worst_cost_tracking_iter.append(np.max([p.cost for p in self.food_sources]))
        pos_diff = [np.abs(np.linalg.norm(p.pos - self.curr_best_agent.pos)) for p in self.food_sources]
        self.pos_diff_mean_iter.append(np.mean(pos_diff))
        self.execution_time_tracking_iter.append(time.time() - self.start_time)

    def clear_prev_results(self):
        self.best_agent = None

        self.food_sources = []
        self.optimum_cost_tracking_iter = []
        self.swarm_cost_tracking_iter = []
        self.curr_best_cost_tracking_iter = []
        self.curr_worst_cost_tracking_iter = []
        self.execution_time_tracking_iter = []
        self.pos_diff_mean_iter = []

        self.start_time = time.time()
        self.scout = []

    def optimize(self):
        self.clear_prev_results()
        self.init_colony()
        self.update_best_solution()

        for i in range(self.n_iter):
            self.employed_bee_phase()
            self.calculate_probabilities()
            self.onlooker_bee_phase()
            self.update_best_solution()
            self.scout.append(self.scout_bee_phase())
            self.memetic_search_phase()
            self.update_best_solution()
            self.iter_track_update()
            # print('Func: {} Iter: {} - Cost: {}'.format(self.objective_function.name, i, self.curr_best_agent.cost))
            self.inet.new_iteration(i, self.best_agent)
