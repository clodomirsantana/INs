import copy
import time

import numpy as np

from networks.InteractionNetwork import InteractionNetwork


class Solution(object):
    def __init__(self, dim):
        nan = float('nan')
        self.pos = [nan for _ in range(dim)]
        self.cost = np.inf
        self.fitness = 0.0
        self.prob = 0.0
        self.trials = 0


class DE(object):
    """
    This is a python implementation of differential evolution
    It assumes an objective_function class is passed in that has the following
    functionality
    data members:
    n              :: The number of parameters
    domain         :: a  list [(low,high)]*n
                     with approximate upper and lower limits for each parameter
    x              :: a place holder for a final solution

    also a function called 'target' is needed.
    This function should take a parameter vector as input and return a the function to be minimized.

    The code below was implemented on the basis of the following sources of information:
    1. http://www.icsi.berkeley.edu/~storn/code.html
    2. http://www.daimi.au.dk/~krink/fec05/articles/JV_ComparativeStudy_CEC04.pdf
    3. http://ocw.mit.edu/NR/rdonlyres/Sloan-School-of-Management/15-099Fall2003/A40397B9-E8FB-4B45-A41B-D1F69218901F/0/ses2_storn_price.pdf


    The developers of the differential evolution method have this advice:
    (taken from ref. 1)

    If you are going to optimize your own objective function with DE, you may try the
    following classical settings for the input file first: Choose method e.g. DE/rand/1/bin,
    set the number of parents NP to 10 times the number of parameters, select weighting
    factor F=0.8, and crossover constant CR=0.9. It has been found recently that selecting
    F from the interval [0.5, 1.0] randomly for each generation or for each difference
    vector, a technique called dither, improves convergence behaviour significantly,
    especially for noisy objective functions. It has also been found that setting CR to a
    low value, e.g. CR=0.2 helps optimizing separable functions since it fosters the search
    along the coordinate axes. On the contrary this choice is not effective if parameter
    dependence is encountered, something which is frequently occuring in real-world optimization
    problems rather than artificial test functions. So for parameter dependence the choice of
    CR=0.9 is more appropriate. Another interesting empirical finding is that rasing NP above,
    say, 40 does not substantially improve the convergence, independent of the number of
    parameters. It is worthwhile to experiment with these suggestions. Make sure that you
    initialize your parameter vectors by exploiting their full numerical range, i.e. if a
    parameter is allowed to exhibit values in the range [-100, 100] it's a good idea to pick
    the initial values from this range instead of unnecessarily restricting diversity.

    Keep in mind that different problems often require different settings for NP, F and CR
    (have a look into the different papers to get a feeling for the settings). If you still
    get misconvergence you might want to try a different method. We mostly use DE/rand/1/... or DE/best/1/... .
    The crossover method is not so important although Ken Price claims that binomial is never
    worse than exponential. In case of misconvergence also check your choice of objective
    function. There might be a better one to describe your problem. Any knowledge that you
    have about the problem should be worked into the objective function. A good objective
    function can make all the difference.

    Note: NP is called population size in the routine below.)
    Note: [0.5,1.0] dither is the default behavior unless f is set to a value other then None.
    """

    def __init__(self, objective_function, population_size=50, cr=0.9, n_cross=1, num_iter=10000, dither_constant=0.4,
                 output_dir=''):
        self.name = "DE"
        self.output_dir = output_dir
        self.dither = dither_constant
        self.objective_function = objective_function
        self.population_size = population_size

        self.cr = cr
        self.n_cross = n_cross
        self.num_iter = num_iter

        self.dim = objective_function.dim
        self.best_agent = Solution(self.dim)
        self.population = []
        self.seeded = False
        self.start_time = None

        for ii in range(self.population_size):
            self.population.append(np.zeros(self.dim))

        self.costs = np.inf * np.ones(self.population_size)

        self.optimum_cost_tracking_iter = []
        self.swarm_cost_tracking_iter = []
        self.curr_best_cost_tracking_iter = []
        self.curr_worst_cost_tracking_iter = []
        self.execution_time_tracking_iter = []
        self.pos_diff_mean_iter = []

        self.inet = []

    def __init_population(self):
        for ii in range(self.dim):
            delta = self.objective_function.max - self.objective_function.min
            offset = self.objective_function.min
            random_values = np.random.sample(self.population_size)
            random_values = random_values * delta + offset
            # now please place these values ni the proper places in the
            # vectors of the population we generated
            for vector, item in zip(self.population, random_values):
                vector[ii] = item
        if self.seeded is not False:
            self.population[0] = self.seeded

        self.inet = InteractionNetwork(self.population_size, directed=True, output_dir=self.output_dir)

    def evaluate_population(self):
        for vector, ii in zip(self.population, range(self.population_size)):
            tmp_score = self.objective_function.evaluate(vector)
            self.costs[ii] = tmp_score
        self.update_best()

    def evolve(self):
        for ii in range(self.population_size):
            rnd = np.random.sample(self.population_size - 1)
            permut = np.argsort(rnd)

            # make parent indices
            i1 = permut[0]
            if i1 >= ii:
                i1 += 1
            i2 = permut[1]
            if i2 >= ii:
                i2 += 1
            i3 = permut[2]
            if i3 >= ii:
                i3 += 1

            x1 = self.population[i1]
            x2 = self.population[i2]
            x3 = self.population[i3]

            use_f = np.random.random() / 2.0 + 0.5
            vi = x1 + use_f * (x2 - x3)

            # prepare the offspring vector please
            rnd = np.random.sample(self.dim)
            permut = np.argsort(rnd)
            test_vector = copy.deepcopy(self.population[ii])
            # first the parameters that sure cross over
            for jj in range(self.dim):
                if jj < self.n_cross:
                    test_vector[permut[jj]] = vi[permut[jj]]
                else:
                    if rnd[jj] > self.cr:
                        test_vector[permut[jj]] = vi[permut[jj]]

            test_score = self.objective_function.evaluate(test_vector)

            # check if the score is lower
            if test_score < self.costs[ii]:
                self.costs[ii] = test_score
                self.population[ii] = test_vector

                self.inet.add_link(ii, i1, np.abs(np.linalg.norm(np.array(self.population[ii]) - self.population[i1])),
                                   [self.costs[ii], self.costs[i1]], [np.array(self.population[ii]), self.population[i1]])
                self.inet.add_link(ii, i2, np.abs(np.linalg.norm(np.array(self.population[ii]) - self.population[i2])),
                                   [self.costs[ii], self.costs[i2]], [np.array(self.population[ii]), self.population[i2]])
                self.inet.add_link(ii, i3, np.abs(np.linalg.norm(np.array(self.population[ii]) - self.population[i3])),
                                   [self.costs[ii], self.costs[i3]], [np.array(self.population[ii]), self.population[i3]])

    def iter_track_update(self):
        self.optimum_cost_tracking_iter.append(self.best_agent.cost)
        self.swarm_cost_tracking_iter.append(self.costs.mean())
        self.curr_best_cost_tracking_iter.append(self.costs.min())
        self.curr_worst_cost_tracking_iter.append(self.costs.max())
        pos_diff = [np.abs(np.linalg.norm(np.array(p) - self.best_agent.pos)) for p in self.population]
        self.pos_diff_mean_iter.append(np.mean(pos_diff))
        self.execution_time_tracking_iter.append(time.time() - self.start_time)

    def update_best(self):
        curr_best = np.argmin(self.costs)
        if self.costs[curr_best] < self.best_agent.cost:
            self.best_agent.cost = self.costs[curr_best]
            self.best_agent.pos = self.population[curr_best]

    def __init_de(self):
        self.best_agent = Solution(self.dim)
        self.swarm_cost_tracking_eval = []
        self.swarm_cost_tracking_iter = []

        self.execution_time_tracking_eval = []
        self.execution_time_tracking_iter = []

        self.start_time = time.time()

    def optimize(self):
        self.__init_de()
        # initialise the population please
        self.__init_population()
        # score the population please
        self.evaluate_population()

        for i in range(self.num_iter):
            self.evolve()
            self.update_best()
            self.iter_track_update()
            self.inet.new_iteration(i, self.best_agent)
            # print('Func: {} Iter: {} - Cost: {}'.format(self.objective_function.name, i, self.best_agent.cost))
        self.inet.save_graphs()

