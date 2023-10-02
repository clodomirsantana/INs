import time
import numpy as np
from scipy.spatial import distance

# This code was based on in the following references:
# [1] "The continuous artificial bee colony algorithm for binary optimization" published in 2015 by
# Mustafa Servet Kiran

# This implementation consider #_employed = #_onlookers = #_food_sources = colony_size / 2
from networks.InteractionNetwork import InteractionNetwork


class Bee(object):
    def __init__(self, dim):
        nan = float('nan')
        self.pos = [nan for _ in range(dim)]
        self.cost = 0.0
        self.fitness = 0.0
        self.train_acc = 0.0
        self.test_acc = 0.0
        self.features = 0.0
        self.prob = 0.0
        self.trials = 0


class ABCBin(object):
    def __init__(self, objective_function, a_lim=5.0, n_iter=1000, colony_size=100, trials_limit=100, output_dir=''):
        self.name = "ABCBin"
        self.output_dir = output_dir
        self.objective_function = objective_function
        self.dim = objective_function.dim
        self.minf = objective_function.minf
        self.maxf = objective_function.maxf
        self.n_iter = n_iter
        self.a_lim = a_lim
        self.start_time = None

        self.best_agent = None
        self.optimum_cost_tracking_iter = []

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

    @staticmethod
    def calculate_fitness(cost):
        if cost >= 0:
            result = 1.0 / (1.0 + cost)
        else:
            result = 1.0 + abs(cost)
        return result

    @staticmethod
    def mapping_abcbin(pos):
        return np.asarray([((round(x % 2)) % 2) for x in pos])

    def clear_prev_data(self):
        self.best_agent = None
        self.food_sources = []
        self.start_time = time.time()

        self.optimum_cost_tracking_iter = []
        self.swarm_cost_tracking_iter = []
        self.curr_best_cost_tracking_iter = []
        self.curr_worst_cost_tracking_iter = []
        self.execution_time_tracking_iter = []
        self.pos_diff_mean_iter = []

    def iter_track_update(self):
        self.optimum_cost_tracking_iter.append(self.best_agent.cost)
        self.swarm_cost_tracking_iter.append(np.mean([p.cost for p in self.food_sources]))
        self.curr_best_cost_tracking_iter.append(np.min([p.cost for p in self.food_sources]))
        self.curr_worst_cost_tracking_iter.append(np.max([p.cost for p in self.food_sources]))
        pos_diff = [np.abs(np.linalg.norm(p.pos - self.best_agent.pos)) for p in self.food_sources]
        self.pos_diff_mean_iter.append(np.mean(pos_diff))
        self.execution_time_tracking_iter.append(time.time() - self.start_time)

    def gen_pos(self):
        return np.random.uniform(-self.a_lim, self.a_lim, self.dim)

    def calculate_probabilities(self):
        sum_fit = 0.0
        for fs in range(self.num_fs):
            sum_fit += self.food_sources[fs].fitness

        for fs in range(self.num_fs):
            self.food_sources[fs].prob = 1 - (self.food_sources[fs].fitness / sum_fit)

    def update_best_solution(self):
        for bee in self.food_sources:
            if bee.cost > self.best_agent.cost:
                self.best_agent.pos = bee.pos
                self.best_agent.cost = bee.cost
                self.best_agent.test_acc = bee.test_acc
                self.best_agent.train_acc = bee.train_acc
                self.best_agent.features = bee.features

    def init_bee(self, pos):
        bee = Bee(self.dim)
        bee.pos = pos
        bee.cost, bee.test_acc, bee.train_acc, bee.features = self.objective_function.evaluate(
            self.mapping_abcbin(bee.pos))
        bee.fitness = self.calculate_fitness(bee.cost)
        return bee

    def init_colony(self):
        self.clear_prev_data()
        self.best_agent = Bee(self.dim)
        self.best_agent.cost = -np.inf

        for i in range(self.num_fs):
            bee = self.init_bee(self.gen_pos())
            self.food_sources.append(bee)

            if bee.cost > self.best_agent.cost:
                self.best_agent.pos = bee.pos
                self.best_agent.cost = bee.cost
                self.best_agent.test_acc = bee.test_acc
                self.best_agent.train_acc = bee.train_acc
                self.best_agent.features = bee.features

        self.inet = InteractionNetwork(self.num_fs, directed=True, output_dir=self.output_dir)

    def employed_bee_phase(self):
        for bee in range(self.num_fs):
            k = list(range(self.num_fs))
            k.remove(bee)
            k = np.random.choice(np.array(k))

            phi = np.random.uniform(-1, 1)
            new_pos = np.copy(self.food_sources[bee].pos)

            j = np.random.choice(range(self.dim))
            new_pos[j] = self.food_sources[bee].pos[j] + phi * (
                    self.food_sources[bee].pos[j] - self.food_sources[k].pos[j])
            cost, test_acc, train_acc, features = self.objective_function.evaluate(self.mapping_abcbin(new_pos))
            fit = self.calculate_fitness(cost)

            if fit < self.food_sources[bee].fitness:
                self.food_sources[bee].pos = new_pos
                self.food_sources[bee].cost = cost
                self.food_sources[bee].fitness = fit
                self.food_sources[bee].trials = 0
                self.food_sources[bee].test_acc = test_acc
                self.food_sources[bee].train_acc = train_acc
                self.food_sources[bee].features = features
                self.inet.add_link(bee, k,
                                   np.abs(distance.hamming(self.mapping_abcbin(self.food_sources[k].pos),
                                                           self.mapping_abcbin(self.food_sources[bee].pos))),
                                   [self.food_sources[bee].cost, self.food_sources[k].cost],
                                   [self.mapping_abcbin(self.food_sources[bee].pos),
                                    self.mapping_abcbin(self.food_sources[k].pos)])
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

                new_pos = np.copy(self.food_sources[s].pos)
                new_pos[j] = new_pos[j] + phi * (new_pos[j] - self.food_sources[k].pos[j])

                cost, test_acc, train_acc, features = self.objective_function.evaluate(self.mapping_abcbin(new_pos))
                fit = self.calculate_fitness(cost)

                if fit < self.food_sources[s].fitness:
                    self.food_sources[s].pos = new_pos
                    self.food_sources[s].cost = cost
                    self.food_sources[s].fitness = fit
                    self.food_sources[s].trials = 0
                    self.food_sources[s].test_acc = test_acc
                    self.food_sources[s].train_acc = train_acc
                    self.food_sources[s].features = features
                    self.inet.add_link(s, k,
                                       np.abs(np.linalg.norm(
                                           self.mapping_abcbin(self.food_sources[k].pos) - self.mapping_abcbin(
                                               self.food_sources[s].pos))),
                                       [self.food_sources[s].cost, self.food_sources[k].cost],
                                       [self.mapping_abcbin(self.food_sources[s].pos),
                                        self.mapping_abcbin(self.food_sources[k].pos)])
                else:
                    self.food_sources[s].trials += 1

    def get_max_trial(self):
        max_ = 0
        for fs in range(self.num_fs):
            if self.food_sources[fs].trials > self.food_sources[max_].trials:
                max_ = fs
        return max_

    def scout_bee_phase(self):
        max_ = self.get_max_trial()

        if self.food_sources[max_].trials >= self.trials_limit:
            pos = self.gen_pos()
            self.food_sources[max_] = self.init_bee(pos)
            self.food_sources[max_].trials = 0
            self.food_sources[max_].cost, self.food_sources[max_].test_acc, self.food_sources[max_].train_acc, \
                self.food_sources[max_].features = self.objective_function.evaluate(pos)

    def optimize(self):
        self.init_colony()
        self.update_best_solution()

        for i in range(self.n_iter):
            self.employed_bee_phase()
            self.calculate_probabilities()
            self.onlooker_bee_phase()
            self.update_best_solution()
            self.scout_bee_phase()
            self.iter_track_update()
            # print('LOG: Iter: {} - Cost: {} '.format(i, self.best_agent.cost))
            self.best_agent.pos = self.mapping_abcbin(self.best_agent.pos)
            self.inet.new_iteration(i, self.best_agent)

        self.inet.save_graphs()
