from algorithms.GA import GA
from problems.ObjectiveFunction import *
from utils.SearchSpaceInitializer import UniformSSInitializer
from utils.Utils import *


def main():
    os.chdir('..')
    num_exec = 30
    population_size = 100
    parents_perc = 0.1
    n_iter = 500
    mr = 0.05
    kb = 0.1

    search_space_initializer = UniformSSInitializer()
    out_dir = "results/"
    funcs = [Sphere]
    dims = [100]

    for func in funcs:
        for d in dims:
            for run in range(num_exec):
                problem = func(d)
                output_dir = out_dir + "{}/{}/Dim_{}/Exec_{}/".format("GA", problem.name, problem.dim, run + 1)

                if not os.path.isdir(output_dir):
                    os.makedirs(output_dir)

                opt = GA(objective_function=problem, search_space_initializer=search_space_initializer,
                         population_size=population_size, n_iter=n_iter, mutation_rate=mr,
                         keep_best=kb, parents_portion=parents_perc, output_dir=output_dir)
                run_experiments(opt, run, output_dir)


if __name__ == '__main__':
    main()
