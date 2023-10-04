from deap.benchmarks import sphere, rosenbrock, rastrigin, griewank, ackley, himmelblau, schaffer, schwefel


class ObjectiveFunction(object):
    def __init__(self, name, dim, minf, maxf, rot=None, trans=None):
        self.name = name
        self.dim = dim
        self.min = minf
        self.max = maxf
        self.rot = rot
        self.trans = trans
        self.is_feature_selection = False

    def evaluate(self, x):
        pass


class Sphere(ObjectiveFunction):
    def __init__(self, dim):
        super(Sphere, self).__init__('Sphere', dim, -100, 100)
        self.func = sphere

    def evaluate(self, x):
        return self.func(x)[0]


class Ackley(ObjectiveFunction):
    def __init__(self, dim):
        super(Ackley, self).__init__('Ackley', dim, -15, 30)
        self.func = ackley

    def evaluate(self, x):
        return self.func(x)[0]


class Griewank(ObjectiveFunction):
    def __init__(self, dim):
        super(Griewank, self).__init__('Griewank', dim, -600, 600)
        self.func = griewank

    def evaluate(self, x):
        return self.func(x)[0]


class Himmelblau(ObjectiveFunction):
    def __init__(self, dim):
        super(Himmelblau, self).__init__('Himmelblau', dim, -6, 6)
        self.func = himmelblau

    def evaluate(self, x):
        return self.func(x)[0]


class Rastrigin(ObjectiveFunction):
    def __init__(self, dim):
        super(Rastrigin, self).__init__('Rastrigin', dim, -5.12, 5.12)
        self.func = rastrigin

    def evaluate(self, x):
        return self.func(x)[0]


class Rosenbrock(ObjectiveFunction):
    def __init__(self, dim):
        super(Rosenbrock, self).__init__('Rosenbrock', dim, -100, 100)
        self.func = rosenbrock

    def evaluate(self, x):
        return self.func(x)[0]


class Schaffer(ObjectiveFunction):
    def __init__(self, dim):
        super(Schaffer, self).__init__('Schaffer', dim, -100, 100)
        self.func = schaffer

    def evaluate(self, x):
        return self.func(x)[0]


class Schwefel(ObjectiveFunction):
    def __init__(self, dim):
        super(Schwefel, self).__init__('Schwefel', dim, -500, 500)
        self.func = schwefel

    def evaluate(self, x):
        return self.func(x)[0]
