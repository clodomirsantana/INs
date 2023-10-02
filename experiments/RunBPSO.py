from algorithms.BPSO import BPSO
from problems.BinaryProblems import *
from utils.Utils import run_experiments


def main():
    os.chdir('..')
    lb_w = 0.1
    up_w = 0.9
    num_exec = 30
    pop_size = 100
    num_iter = 500
    c1 = (0.72984 * 2.05)
    c2 = (0.72984 * 2.05)

    out_dir = "results/"
    funcs = [ZeroOneKnapsack]
    dims = [10]

    for func in funcs:
        for d in dims:
            for run in range(num_exec):
                problem = func(d)
                output_dir = out_dir + "{}/{}/Dim_{}/Exec_{}/".format("BPSO", problem.name, problem.dim, run + 1)

                if not os.path.isdir(output_dir):
                    os.makedirs(output_dir)

                opt = BPSO(objective_function=problem, pop_size=pop_size, max_iter=num_iter, lb_w=lb_w, up_w=up_w,
                           c1=c1, c2=c2, v_max=100000, maximize=True, output_dir=output_dir)
                run_experiments(opt, run, output_dir)


if __name__ == '__main__':
    main()
