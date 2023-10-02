from algorithms.ABC import ABC
from problems.ObjectiveFunction import *
from utils.SearchSpaceInitializer import UniformSSInitializer
from utils.Utils import *


def main():
    os.chdir('..')
    num_exec = 30
    colony_size = 200
    num_iterations = 500
    trials_limit = 100
    out_dir = "results/"

    dims = [50]
    funcs = [Sphere]
    search_space_initializer = UniformSSInitializer()

    for func in funcs:
        for d in dims:
            for run in range(num_exec):
                problem = func(d)
                output_dir = out_dir + "{}/{}/Dim_{}/Exec_{}/".format("ABC", problem.name, problem.dim, run + 1)

                if not os.path.isdir(output_dir):
                    os.makedirs(output_dir)

                opt = ABC(objective_function=problem, search_space_initializer=search_space_initializer,
                          n_iter=num_iterations, colony_size=colony_size, trials_limit=trials_limit,
                          output_dir=output_dir)
                run_experiments(opt, run, output_dir)


if __name__ == '__main__':
    main()
