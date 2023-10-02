from __future__ import division
import random
import time

import numpy as np
from scipy.spatial import distance
from networks.InteractionNetwork import InteractionNetwork


class Genome(object):
    BINARY_BASE = 2

    def __init__(self, dim):
        self.pos = Genome.generate_random_genome(dim)
        self.cross_op = None
        self.mutation_op = None
        self.cost = -10 ** 5
        self.train_acc = 0.0
        self.test_acc = 0.0
        self.features = 0.0

    def mutate(self, prob):
        if random.random() <= prob:
            if not self.mutation_op:
                idx = np.random.choice(range(len(self.pos)))
                self.pos[idx] = 1 - self.pos[idx]
            else:
                self.pos = self.mutation_op(self.pos)

    def crossover(self, ga, gb):
        if not self.cross_op:
            idx = np.random.choice(range(len(ga)))
            self.pos[:idx] = ga[:idx]
            self.pos[idx:] = gb[idx:]
        else:
            self.pos = self.cross_op(ga, gb)

    @staticmethod
    def generate_random_genome(dim):
        return np.random.randint(Genome.BINARY_BASE, size=dim)


class BGA(object):
    def __init__(self, objective_function, pop_size=1000, mutation_rate=0.9, cross_rate=1.0,
                 max_iter=5000, maximize=True, elitism=True, output_dir=''):
        self.name = "BGA"
        self.output_dir = output_dir
        self.dim = objective_function.dim
        self.objective_function = objective_function
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.cross_rate = cross_rate
        self.max_iter = max_iter
        self.elitism = elitism
        self.cross_op = None
        self.mutation_op = None
        self.op = max if maximize else min
        self.nop = min if maximize else max

        self.best_agent = None
        self.start_time = None

        self.optimum_cost_tracking_iter = []
        self.swarm_cost_tracking_iter = []
        self.curr_best_cost_tracking_iter = []
        self.curr_worst_cost_tracking_iter = []
        self.execution_time_tracking_iter = []
        self.pos_diff_mean_iter = []

        self.inet = []
        self.genes = []

    def __initialize_population(self):
        self.genes = []
        for _ in range(self.pop_size):
            individual = Genome(dim=self.dim)
            self.genes.append(individual)

        if self.cross_op:
            for g in self.genes:
                self.__set_cross_op(self.cross_op, g)

        if self.mutation_op:
            for g in self.genes:
                self.__set_mutation_op(self.mutation_op, g)

        self.start_time = time.time()

        self.best_agent = np.random.choice(self.genes)
        self.inet = InteractionNetwork(self.pop_size, directed=True, output_dir=self.output_dir)

        self.optimum_cost_tracking_iter = []
        self.swarm_cost_tracking_iter = []
        self.curr_best_cost_tracking_iter = []
        self.curr_worst_cost_tracking_iter = []
        self.execution_time_tracking_iter = []
        self.pos_diff_mean_iter = []

    def __iter_track_update(self):
        self.optimum_cost_tracking_iter.append(self.best_agent.cost)
        self.swarm_cost_tracking_iter.append(np.mean([p.cost for p in self.genes]))
        self.curr_best_cost_tracking_iter.append(np.min([p.cost for p in self.genes]))
        self.curr_worst_cost_tracking_iter.append(np.max([p.cost for p in self.genes]))
        pos_diff = [np.abs(np.linalg.norm(p.pos - self.best_agent.pos)) for p in self.genes]
        self.pos_diff_mean_iter.append(np.mean(pos_diff))
        self.execution_time_tracking_iter.append(time.time() - self.start_time)

    def best(self):
        return self.best_agent.pos

    def set_cross_operator(self, op):
        self.cross_op = op

    def set_mutation_operator(self, op):
        self.mutation_op = op

    def __evaluate_population(self):
        for g in self.genes:
            self.__evaluate(self.objective_function, g)
        self.best_agent = self.op(self.genes, key=lambda g: g.cost)

    def __get_fittest(self):
        if self.elitism:
            self.genes.remove(self.best_agent)

    def __retrieve(self):
        if self.best_agent not in self.genes:
            self.genes.append(self.best_agent)

    def __induce_mutation(self):
        for g in self.genes:
            self.__mutation(self.mutation_rate, g)

    def __induce_cross_over(self):
        new_gs = []
        for _ in range(int(0.05 * self.pop_size)):  # Replace up to 10% of the worst genes in the population
            if random.random() <= self.cross_rate:
                inferior_gene1 = self.nop(self.genes, key=lambda g: g.cost)
                self.genes.remove(inferior_gene1)
                inferior_gene2 = self.nop(self.genes, key=lambda g: g.cost)
                self.genes.remove(inferior_gene2)

                idx1, idx2 = self.__pick_parents()
                inferior_gene1.crossover(idx1.pos, idx2.pos)
                self.__evaluate(self.objective_function, inferior_gene1)
                new_gs.append(inferior_gene1)
                self.inet.add_link(inferior_gene1, idx1, np.abs(distance.hamming(idx1.pos, inferior_gene1.pos)),
                                   [inferior_gene1.cost, idx1.cost], [inferior_gene1.pos, idx1.pos])
                self.inet.add_link(inferior_gene1, idx2, np.abs(distance.hamming(idx2.pos, inferior_gene1.pos)),
                                   [inferior_gene1.cost, idx2.cost], [inferior_gene1.pos, idx2.pos])

                inferior_gene2.crossover(idx2.pos, idx1.pos)
                self.__evaluate(self.objective_function, inferior_gene2)
                new_gs.append(inferior_gene2)
                self.inet.add_link(inferior_gene2, idx1, np.abs(distance.hamming(idx1.pos, inferior_gene2.pos)),
                                   [inferior_gene2.cost, idx1.cost], [inferior_gene2.pos, idx1.pos])
                self.inet.add_link(inferior_gene2, idx2, np.abs(distance.hamming(idx2.pos, inferior_gene2.pos)),
                                   [inferior_gene2.cost, idx2.cost], [inferior_gene2.pos, idx2.pos])
        for g in new_gs:
            self.genes.append(g)

    def __pick_parents(self):
        trials = 0
        idx1 = BGA.__roulette_selection(self.genes)
        idx2 = BGA.__roulette_selection(self.genes)
        while not idx2 != idx1 and trials < 1000:
            idx2 = BGA.__roulette_selection(self.genes)
            trials += 1
        return idx1, idx2

    @staticmethod
    def __set_cross_op(op, genome):
        genome.cross_op = op

    @staticmethod
    def __set_mutation_op(op, genome):
        genome.mutation_op = op

    def __evaluate(self, fitness, genome):
        genome.cost, genome.test_acc, genome.train_acc, genome.features = fitness.evaluate(genome.pos)

    @staticmethod
    def __mutation(prob, genome):
        genome.mutate(prob)

    @staticmethod
    def __roulette_selection(genes):
        max_ = sum(gene.cost for gene in genes)
        pick = random.uniform(0, max_)
        current = 0
        for gene in genes:
            current += gene.cost
            if current > pick:
                return gene

    def optimize(self):
        self.__initialize_population()
        self.__evaluate_population()

        for i in range(self.max_iter):
            self.__get_fittest()
            self.__induce_cross_over()
            self.__induce_mutation()
            self.__retrieve()
            self.__evaluate_population()
            self.__iter_track_update()
            # print('LOG: Iter: {} - Cost: {}'.format(i, self.best_agent.cost))
            self.inet.new_iteration(i, self.best_agent)
        self.inet.save_graphs()
