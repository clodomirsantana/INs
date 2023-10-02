import copy
import time

import numpy as np

from networks.InteractionNetwork import InteractionNetwork


# This code was based on in the following references:
# [1] "Genetic Algorithms" published in 2005 by John H. Holland


class Chromosome(object):
    def __init__(self, dim):
        nan = float('nan')
        self.pos = [nan for _ in range(dim)]
        self.cost = np.nan
        self.prob = 0


class GA(object):
    def __init__(self, objective_function, search_space_initializer, population_size=30, n_iter=1000,
                 parents_portion=0.3, mutation_rate=0.05, keep_best=0.1, output_dir=''):
        self.name = "GA"
        self.output_dir = output_dir
        self.objective_function = objective_function
        self.search_space_initializer = search_space_initializer
        self.start_time = None

        self.dim = objective_function.dim
        self.minf = objective_function.min
        self.maxf = objective_function.max

        self.population_size = population_size
        self.num_parents = int(parents_portion * population_size)
        self.n_iter = n_iter
        self.mutation_rate = mutation_rate

        # Variables that store of state of GA
        self.population = []
        self.parents = []
        self.next_pop = []
        self.keep_best = int(keep_best * population_size)

        # gbest of the population
        self.best_agent = Chromosome(self.dim)
        self.best_agent.cost = np.inf

        self.optimum_cost_tracking_iter = []
        self.swarm_cost_tracking_iter = []
        self.curr_best_cost_tracking_iter = []
        self.curr_worst_cost_tracking_iter = []
        self.execution_time_tracking_iter = []
        self.pos_diff_mean_iter = []

        self.inet = []
        # used on the Boltzmann selection
        self.t_init = 100
        # used on the Shrink mutation
        self.shrink_factor = 0.1 * (self.maxf - self.minf)

    def __init_population(self):
        self.inet = InteractionNetwork(self.population_size, directed=True, output_dir=self.output_dir)
        self.best_agent = Chromosome(self.dim)
        self.best_agent.cost = np.inf

        x = self.search_space_initializer.sample(self.objective_function, self.population_size)

        for i in range(self.population_size):
            chromosome = Chromosome(self.dim)
            chromosome.pos = x[i]
            chromosome.cost = self.objective_function.evaluate(chromosome.pos)
            self.population.append(chromosome)

        self.optimum_cost_tracking_iter.append(self.best_agent.cost)

    # Restart the GA
    def __init_ga(self):
        self.best_agent = None
        self.population = []

        self.optimum_cost_tracking_iter = []
        self.swarm_cost_tracking_iter = []
        self.curr_best_cost_tracking_iter = []
        self.curr_worst_cost_tracking_iter = []
        self.execution_time_tracking_iter = []
        self.pos_diff_mean_iter = []

        self.start_time = time.time()
        self.t_init = 100
        self.shrink_factor = 0.1 * (self.maxf - self.minf)

    def iter_track_update(self):
        self.optimum_cost_tracking_iter.append(self.best_agent.cost)
        self.swarm_cost_tracking_iter.append(np.mean([p.cost for p in self.population]))
        self.curr_best_cost_tracking_iter.append(np.min([p.cost for p in self.population]))
        self.curr_worst_cost_tracking_iter.append(np.max([p.cost for p in self.population]))
        pos_diff = [np.abs(np.linalg.norm(np.array(p.pos) - self.best_agent.pos)) for p in self.population]
        self.pos_diff_mean_iter.append(np.mean(pos_diff))
        self.execution_time_tracking_iter.append(time.time() - self.start_time)

    def selection_best(self):
        ranked = sorted(self.population, key=lambda gene: gene.cost)[:self.num_parents]
        self.parents = [list(self.population).index(b) for b in ranked]
        self.get_next_pop()

    def selection_boltzmann(self, curr_iter):
        self.parents = []
        alpha = 0.25
        fit_mean = np.mean([x.cost for x in self.population])
        k = 1 + 100 * (curr_iter / self.n_iter)
        t = self.t_init * ((1 - alpha) ** k)

        while len(self.parents) < self.num_parents:
            rand_p = np.random.choice(range(self.population_size))
            if (self.population[rand_p].cost < fit_mean) and (rand_p not in self.parents):
                self.parents.append(rand_p)
            elif (np.exp(-(self.population[rand_p].cost - fit_mean) / t) > np.random.uniform()) and \
                    (rand_p not in self.parents):
                self.parents.append(rand_p)
        self.get_next_pop()

    def selection_roulette_wheel(self):
        # Computes the totality of the population fitness
        best_idxs = []
        total = sum([np.abs(chromosome.cost) for chromosome in self.population])
        i, n = 0, self.num_parents
        w, v = self.population[0].cost, self.population[0]
        while n:
            x = total * (1 - np.random.uniform() ** (1.0 / n))
            total -= x
            while x > (total - w):
                x -= w
                i = (i + 1) % self.population_size
                w, v = self.population[i].cost, self.population[i]
            w -= x
            best_idxs.append(v)
            n -= 1

        self.parents = [list(self.population).index(b) for b in best_idxs]
        self.get_next_pop()

    def selection_binary_tournament(self):
        res = []
        while len(res) < self.num_parents:
            t1 = np.random.choice(range(self.population_size))
            t2 = np.random.choice(range(self.population_size))
            if self.population[t1].cost < self.population[t2].cost:
                res.append(t1)
        self.parents = res
        self.get_next_pop()

    def selection_random(self):
        self.parents = np.random.choice(range(self.population_size), size=self.num_parents)
        self.get_next_pop()

    def crossover_single_point(self):
        children = []
        target_children_size = self.population_size - len(self.next_pop)

        while len(children) < target_children_size:
            crossover_point = np.random.randint(1, self.dim - 1)
            father = self.population[np.random.choice(self.parents)]
            mother = self.population[np.random.choice(self.parents)]

            pos1 = np.concatenate((father.pos[:crossover_point], mother.pos[crossover_point:]))
            pos2 = np.concatenate((mother.pos[:crossover_point], father.pos[crossover_point:]))

            child1 = Chromosome(self.dim)
            child2 = Chromosome(self.dim)

            child1.pos = copy.deepcopy(pos1)
            child2.pos = copy.deepcopy(pos2)

            child1.cost = self.objective_function.evaluate(np.asarray(pos1))
            child2.cost = self.objective_function.evaluate(np.asarray(pos2))

            children.append(child1)

            self.inet.add_link(child1, father, np.abs(np.linalg.norm(np.array(father.pos) - child1.pos)),
                               [child1.cost, father.cost], [child1.pos, father.pos])
            self.inet.add_link(child1, mother, np.abs(np.linalg.norm(np.array(mother.pos) - child1.pos)),
                               [child1.cost, mother.cost], [child1.pos, mother.pos])

            if len(children) < target_children_size:
                children.append(child2)

                self.inet.add_link(child2, father, np.abs(np.linalg.norm(np.array(father.pos) - child2.pos)),
                                   [child2.cost, father.cost], [child2.pos, father.pos])
                self.inet.add_link(child2, mother, np.abs(np.linalg.norm(np.array(mother.pos) - child2.pos)),
                                   [child2.cost, mother.cost], [child2.pos, mother.pos])

        return children

    def crossover_double_point(self):
        children = []
        target_children_size = self.population_size - len(self.next_pop)

        while len(children) < target_children_size:
            cross_pt1, cross_pt2 = sorted(np.random.randint(1, self.dim - 1, 2))
            father = self.population[np.random.choice(self.parents)]
            mother = self.population[np.random.choice(self.parents)]

            pos1 = np.concatenate((father.pos[:cross_pt1], mother.pos[cross_pt1:cross_pt2], father.pos[cross_pt2:]))
            pos2 = np.concatenate((mother.pos[:cross_pt1], father.pos[cross_pt1:cross_pt2], mother.pos[cross_pt2:]))

            child1 = Chromosome(self.dim)
            child2 = Chromosome(self.dim)

            child1.pos = copy.deepcopy(pos1)
            child2.pos = copy.deepcopy(pos2)

            child1.cost = self.objective_function.evaluate(np.asarray(pos1))
            child2.cost = self.objective_function.evaluate(np.asarray(pos2))

            children.append(child1)

            self.inet.add_link(child1, father, np.abs(np.linalg.norm(np.array(father.pos) - child1.pos)),
                               [child1.cost, father.cost], [child1.pos, father.pos])
            self.inet.add_link(child1, mother, np.abs(np.linalg.norm(np.array(mother.pos) - child1.pos)),
                               [child1.cost, mother.cost], [child1.pos, mother.pos])

            if len(children) < target_children_size:
                children.append(child2)

                self.inet.add_link(child2, father, np.abs(np.linalg.norm(np.array(father.pos) - child2.pos)),
                                   [child2.cost, father.cost], [child2.pos, father.pos])
                self.inet.add_link(child2, mother, np.abs(np.linalg.norm(np.array(mother.pos) - child2.pos)),
                                   [child2.cost, mother.cost], [child2.pos, mother.pos])

        return children

    def crossover_uniform(self):
        children = []
        target_children_size = self.population_size - len(self.next_pop)

        while len(children) < target_children_size:
            father = self.population[np.random.choice(self.parents)]
            mother = self.population[np.random.choice(self.parents)]

            child1 = Chromosome(self.dim)
            child2 = Chromosome(self.dim)

            for d in range(self.dim):
                if np.random.uniform() > 0.5:
                    child1.pos[d] = father.pos[d]
                    child2.pos[d] = mother.pos[d]
                else:
                    child2.pos[d] = father.pos[d]
                    child1.pos[d] = mother.pos[d]

            child1.cost = self.objective_function.evaluate(child1.pos)
            child2.cost = self.objective_function.evaluate(child2.pos)

            children.append(child1)

            self.inet.add_link(child1, father, np.abs(np.linalg.norm(np.array(father.pos) - child1.pos)),
                               [child1.cost, father.cost], [child1.pos, father.pos])
            self.inet.add_link(child1, mother, np.abs(np.linalg.norm(np.array(mother.pos) - child1.pos)),
                               [child1.cost, mother.cost], [child1.pos, mother.pos])

            if len(children) < target_children_size:
                children.append(child2)

                self.inet.add_link(child2, father, np.abs(np.linalg.norm(np.array(father.pos) - child2.pos)),
                                   [child2.cost, father.cost], [child2.pos, father.pos])
                self.inet.add_link(child2, mother, np.abs(np.linalg.norm(np.array(mother.pos) - child2.pos)),
                                   [child2.cost, mother.cost], [child2.pos, mother.pos])

        return children

    def crossover_laplace(self):
        children = []
        target_children_size = self.population_size - len(self.next_pop)

        while len(children) < target_children_size:
            father = self.population[np.random.choice(self.parents)]
            mother = self.population[np.random.choice(self.parents)]

            child1 = Chromosome(self.dim)
            child2 = Chromosome(self.dim)

            for d in range(self.dim):
                a, b = 0.0, 0.25
                u, u_prime = np.random.uniform(size=2)
                beta = (a - b * np.log(u)) if u_prime <= 0.5 else (a + b * np.log(u))

                child1.pos[d] = father.pos[d] + beta * (np.abs(father.pos[d] - mother.pos[d]))
                child2.pos[d] = mother.pos[d] + beta * (np.abs(father.pos[d] - mother.pos[d]))

            child1.cost = self.objective_function.evaluate(child1.pos)
            child2.cost = self.objective_function.evaluate(child2.pos)

            children.append(child1)

            self.inet.add_link(child1, father, np.abs(np.linalg.norm(np.array(father.pos) - child1.pos)),
                               [child1.cost, father.cost], [child1.pos, father.pos])
            self.inet.add_link(child1, mother, np.abs(np.linalg.norm(np.array(mother.pos) - child1.pos)),
                               [child1.cost, mother.cost], [child1.pos, mother.pos])

            if len(children) < target_children_size:
                children.append(child2)

                self.inet.add_link(child2, father, np.abs(np.linalg.norm(np.array(father.pos) - child2.pos)),
                                   [child2.cost, father.cost], [child2.pos, father.pos])
                self.inet.add_link(child2, mother, np.abs(np.linalg.norm(np.array(mother.pos) - child2.pos)),
                                   [child2.cost, mother.cost], [child2.pos, mother.pos])

        return children

    def crossover_three_parent(self):
        children = []
        target_children_size = self.population_size - len(self.next_pop)

        while len(children) < target_children_size:
            parents = [self.population[p] for p in np.random.choice(self.parents, 3)]
            cross_pt1, cross_pt2 = sorted(np.random.randint(1, self.dim - 1, 2))

            pos1 = np.concatenate(
                (parents[0].pos[:cross_pt1], parents[1].pos[cross_pt1:cross_pt2], parents[2].pos[cross_pt2:]))
            pos2 = np.concatenate(
                (parents[1].pos[:cross_pt1], parents[2].pos[cross_pt1:cross_pt2], parents[0].pos[cross_pt2:]))
            pos3 = np.concatenate(
                (parents[2].pos[:cross_pt1], parents[0].pos[cross_pt1:cross_pt2], parents[1].pos[cross_pt2:]))

            child1 = Chromosome(self.dim)
            child2 = Chromosome(self.dim)
            child3 = Chromosome(self.dim)

            child1.pos = copy.deepcopy(pos1)
            child2.pos = copy.deepcopy(pos2)
            child3.pos = copy.deepcopy(pos3)

            child1.cost = self.objective_function.evaluate(np.asarray(pos1))
            child2.cost = self.objective_function.evaluate(np.asarray(pos2))
            child3.cost = self.objective_function.evaluate(np.asarray(pos3))

            children.append(child1)

            self.inet.add_link(child1, parents[0], np.abs(np.linalg.norm(np.array(parents[0].pos) - child1.pos)),
                               [child1.cost, parents[0].cost], [child1.pos, parents[0].pos])
            self.inet.add_link(child1, parents[1], np.abs(np.linalg.norm(np.array(parents[1].pos) - child1.pos)),
                               [child1.cost, parents[1].cost], [child1.pos, parents[1].pos])
            self.inet.add_link(child1, parents[2], np.abs(np.linalg.norm(np.array(parents[2].pos) - child1.pos)),
                               [child1.cost, parents[2].cost], [child1.pos, parents[2].pos])

            if len(children) < target_children_size:
                children.append(child2)

                self.inet.add_link(child2, parents[0], np.abs(np.linalg.norm(np.array(parents[0].pos) - child2.pos)),
                                   [child2.cost, parents[0].cost], [child2.pos, parents[0].pos])
                self.inet.add_link(child2, parents[1], np.abs(np.linalg.norm(np.array(parents[1].pos) - child2.pos)),
                                   [child2.cost, parents[1].cost], [child2.pos, parents[1].pos])
                self.inet.add_link(child2, parents[2], np.abs(np.linalg.norm(np.array(parents[2].pos) - child2.pos)),
                                   [child2.cost, parents[2].cost], [child2.pos, parents[2].pos])
                if len(children) < target_children_size:
                    children.append(child3)

                    self.inet.add_link(child3, parents[0],
                                       np.abs(np.linalg.norm(np.array(parents[0].pos) - child3.pos)),
                                       [child3.cost, parents[0].cost], [child3.pos, parents[0].pos])
                    self.inet.add_link(child3, parents[1],
                                       np.abs(np.linalg.norm(np.array(parents[1].pos) - child3.pos)),
                                       [child3.cost, parents[1].cost], [child3.pos, parents[1].pos])
                    self.inet.add_link(child3, parents[2],
                                       np.abs(np.linalg.norm(np.array(parents[2].pos) - child3.pos)),
                                       [child3.cost, parents[2].cost], [child3.pos, parents[2].pos])

        return children

    def mutation_uniform(self, children):
        for c in children:
            for mutation_point in range(self.dim):
                if np.random.uniform() <= self.mutation_rate:
                    c.pos[mutation_point] += np.random.uniform(-1, 1)
                    c.cost = self.objective_function.evaluate(np.asarray(c.pos))
        self.population = np.append(self.next_pop, children)

    def mutation_gaussian(self, children):
        mu, sigma = 0, 0.2
        for c in children:
            for mutation_point in range(self.dim):
                if np.random.uniform() <= self.mutation_rate:
                    c.pos[mutation_point] += np.random.normal(mu, sigma)
                    c.cost = self.objective_function.evaluate(np.asarray(c.pos))
        self.population = np.append(self.next_pop, children)

    def mutation_non_uniform(self, children, i):
        def _delta_func(y):
            return y * (1 - np.random.uniform() ** ((1 - i / self.n_iter) ** 0.25))

        for c in children:
            for mutation_point in range(self.dim):
                if np.random.uniform() <= self.mutation_rate:
                    if np.random.uniform() <= 0.5:
                        c.pos[mutation_point] = c.pos[mutation_point] + _delta_func(self.maxf - c.pos[mutation_point])
                    else:
                        c.pos[mutation_point] = c.pos[mutation_point] - _delta_func(c.pos[mutation_point] - self.minf)
                    c.cost = self.objective_function.evaluate(np.asarray(c.pos))
        self.population = np.append(self.next_pop, children)

    def mutation_mptm(self, children):
        for c in children:
            for mutation_point in range(self.dim):
                if np.random.uniform() <= self.mutation_rate:
                    r = np.random.uniform()
                    t = (c.pos[mutation_point] - self.minf) / (self.maxf - c.pos[mutation_point])
                    t_prime = t
                    if r < t:
                        t_prime = t - t * (((t - r) / t) ** 4)
                    elif r > t:
                        t_prime = t + (1 - t) * (((r - t) / (1 - t)) ** 4)

                    c.pos[mutation_point] = (1 - t_prime) * self.minf + t_prime * self.maxf
                    c.cost = self.objective_function.evaluate(np.asarray(c.pos))
        self.population = np.append(self.next_pop, children)

    def mutation_shrink(self, children, i):
        self.shrink_factor *= (1 - (i / self.n_iter))
        for c in children:
            for mutation_point in range(self.dim):
                if np.random.uniform() <= self.mutation_rate:
                    c.pos[mutation_point] += self.shrink_factor * (np.random.normal(0, 0.15) * np.sqrt(self.shrink_factor))
                    c.cost = self.objective_function.evaluate(np.asarray(c.pos))
        self.population = np.append(self.next_pop, children)

    def update_best_sol(self):
        for c in self.population:
            if c.cost < self.best_agent.cost:
                self.best_agent.pos = copy.deepcopy(c.pos)
                self.best_agent.cost = c.cost

    def get_next_pop(self):
        self.next_pop = sorted(self.population, key=lambda gene: gene.cost)[:self.keep_best]
        best_idxs = [list(self.population).index(b) for b in self.next_pop]
        for p in self.parents:
            if p not in best_idxs:
                self.next_pop.append(self.population[p])

    def optimize(self):
        self.__init_ga()
        self.__init_population()

        for i in range(self.n_iter):
            self.selection_roulette_wheel()
            children = self.crossover_double_point()
            self.mutation_gaussian(children)
            self.update_best_sol()
            self.iter_track_update()
            self.inet.new_iteration(i, self.best_agent)
            # print('Func: {} Iter: {} - Cost: {}'.format(self.objective_function.name, i, self.best_agent.cost))
        self.inet.save_graphs()
