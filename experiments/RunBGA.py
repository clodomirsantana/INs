from algorithms.BGA import BGA
from problems.BinaryProblems import *
from utils.Utils import run_experiments


def main():
    os.chdir('..')
    num_exec = 30
    pop_size = 100
    num_iter = 500
    mutation = 0.66
    cross = 0.9

    out_dir = "results/"
    funcs = [ZeroOneKnapsack]
    dims = [100]

    for func in funcs:
        for d in dims:
            for run in range(num_exec):
                problem = func(d)
                output_dir = out_dir + "{}/{}/Dim_{}/Exec_{}/".format("BGA", problem.name, problem.dim, run + 1)

                if not os.path.isdir(output_dir):
                    os.makedirs(output_dir)

                opt = BGA(problem, pop_size=pop_size, mutation_rate=mutation, cross_rate=cross, max_iter=num_iter,
                          output_dir=output_dir)
                run_experiments(opt, run, output_dir)


if __name__ == '__main__':
    main()
