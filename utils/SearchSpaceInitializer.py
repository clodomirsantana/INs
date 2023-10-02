import numpy as np


class SearchSpaceInitializer(object):
    def sample(self, objective_function, n):
        pass


class UniformSSInitializer(SearchSpaceInitializer):
    def sample(self, objective_function, n):
        x = np.zeros((n, objective_function.dim))
        for i in range(n):
            x[i] = np.random.uniform(objective_function.min, objective_function.max, objective_function.dim)
        return x


class OneQuarterDimWiseSSInitializer(SearchSpaceInitializer):
    def sample(self, objective_function, n):
        min_init_fb = objective_function.max - ((1.0 / 4.0) * (objective_function.max - objective_function.min))
        max_init_fb = objective_function.max

        x = np.zeros((n, objective_function.dim))
        for i in range(n):
            x[i] = np.random.uniform(min_init_fb, max_init_fb, objective_function.dim)
        return x
