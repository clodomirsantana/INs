from algorithms.ABCBin import ABCBin
from problems.BinaryProblems import *
from utils.Utils import run_experiments


def main():
    os.chdir('..')
    num_exec = 30
    colony_size = 200
    num_iter = 500
    a_lim = 5.0
    trials_limit = 10
    out_dir = "results/"
    funcs = [ZeroOneKnapsack, ZeroMax]
    dims = [10]

    for func in funcs:
        for d in dims:
            for run in range(num_exec):
                problem = func(d)
                output_dir = out_dir + "{}/{}/Dim_{}/Exec_{}/".format("ABCBin", problem.name, problem.dim, run + 1)

                if not os.path.isdir(output_dir):
                    os.makedirs(output_dir)

                opt = ABCBin(objective_function=problem, a_lim=a_lim, n_iter=num_iter, colony_size=colony_size,
                             trials_limit=trials_limit, output_dir=output_dir)
                run_experiments(opt, run, output_dir)


if __name__ == '__main__':
    main()
