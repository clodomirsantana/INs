from cec2013lsgo.cec2013 import Benchmark

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


class CECF0(ObjectiveFunction):
    def __init__(self, dim):
        super(CECF0, self).__init__('CEC F0', dim, -100, 100)
        self.func = sphere

    def evaluate(self, x):
        return self.func(x)[0]

class CECF1(ObjectiveFunction):
    def __init__(self, dim):
        super(CECF1, self).__init__('CEC F1', dim, -100, 100)
        self.func = ackley

    def evaluate(self, x):
        return self.func(x)[0]


class CECF11(ObjectiveFunction):
    def __init__(self):
        super(CECF11, self).__init__('CEC F11', 1000, -100, 100)
        self.func = Benchmark().get_function(1)

    def evaluate(self, x):
        return self.func(x.astype(float))


class CECF2(ObjectiveFunction):
    def __init__(self, dim):
        super(CECF2, self).__init__('CEC F2', dim, -5, 5)
        self.func = Benchmark().get_function(2)

    def evaluate(self, x):
        return self.func(x.astype(float))


class CECF3(ObjectiveFunction):
    def __init__(self, dim):
        super(CECF3, self).__init__('CEC F3', dim, -32, 32)
        self.func = Benchmark().get_function(3)

    def evaluate(self, x):
        return self.func(x.astype(float))


class CECF4(ObjectiveFunction):
    def __init__(self, dim):
        super(CECF4, self).__init__('CEC F4', dim, -100, 100)
        self.func = Benchmark().get_function(4)

    def evaluate(self, x):
        return self.func(x.astype(float))


class CECF5(ObjectiveFunction):
    def __init__(self, dim):
        super(CECF5, self).__init__('CEC F5', dim, -5, 5)
        self.func = Benchmark().get_function(5)

    def evaluate(self, x):
        return self.func(x.astype(float))


class CECF6(ObjectiveFunction):
    def __init__(self):
        super(CECF6, self).__init__('CECF6', 1000, -32, 32)
        self.func = Benchmark().get_function(6)

    def evaluate(self, x):
        return self.func(x.astype(float))


class CECF7(ObjectiveFunction):
    def __init__(self):
        super(CECF7, self).__init__('CEC F7', 1000, -100, 100)
        self.func = Benchmark().get_function(7)

    def evaluate(self, x):
        return self.func(x.astype(float))


class CECF8(ObjectiveFunction):
    def __init__(self):
        super(CECF8, self).__init__('CEC F8', 1000, -100, 100)
        self.func = Benchmark().get_function(8)

    def evaluate(self, x):
        return self.func(x.astype(float))


class CECF9(ObjectiveFunction):
    def __init__(self):
        super(CECF9, self).__init__('CEC F9', 1000, -5, 5)
        self.func = Benchmark().get_function(9)

    def evaluate(self, x):
        return self.func(x.astype(float))


class CECF10(ObjectiveFunction):
    def __init__(self):
        super(CECF10, self).__init__('CEC F10', 1000, -32, 32)
        self.func = Benchmark().get_function(10)

    def evaluate(self, x):
        return self.func(x.astype(float))


class CECF11(ObjectiveFunction):
    def __init__(self):
        super(CECF11, self).__init__('CEC F11', 1000, -100, 100)
        self.func = Benchmark().get_function(11)

    def evaluate(self, x):
        return self.func(x.astype(float))


class CECF12(ObjectiveFunction):
    def __init__(self):
        super(CECF12, self).__init__('CEC F12', 1000, -100, 100)
        self.func = Benchmark().get_function(12)

    def evaluate(self, x):
        return self.func(x.astype(float))


class CECF13(ObjectiveFunction):
    def __init__(self):
        super(CECF13, self).__init__('CEC F13', 1000, -100, 100)
        self.func = Benchmark().get_function(13)

    def evaluate(self, x):
        return self.func(x.astype(float))


class CECF14(ObjectiveFunction):
    def __init__(self):
        super(CECF14, self).__init__('CEC F14', 1000, -100, 100)
        self.func = Benchmark().get_function(14)

    def evaluate(self, x):
        return self.func(x.astype(float))


class CECF15(ObjectiveFunction):
    def __init__(self):
        super(CECF15, self).__init__('CEC F15', 1000, -100, 100)
        self.func = Benchmark().get_function(15)

    def evaluate(self, x):
        return self.func(x.astype(float))
