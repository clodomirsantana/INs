from algorithms.DE import DE
from problems.ObjectiveFunction import *
from utils.SearchSpaceInitializer import UniformSSInitializer
from utils.Utils import *


def main():
    os.chdir('..')
    dither_constant = 0.4
    population_size = 100
    num_iter = 500
    num_exec = 30
    n_cross = 1
    cr = 0.9

    out_dir = "results/"
    funcs = [Sphere]
    dims = [50]

    for func in funcs:
        for d in dims:
            for run in range(num_exec):
                problem = func(d)
                output_dir = out_dir + "{}/{}/Dim_{}/Exec_{}/".format("DE", problem.name, problem.dim, run + 1)

                if not os.path.isdir(output_dir):
                    os.makedirs(output_dir)

                opt = DE(problem, population_size, cr, n_cross, num_iter, dither_constant, output_dir=output_dir)
                run_experiments(opt, run, output_dir)


if __name__ == '__main__':
    main()
