from algorithms.BCSO import BCSO
from problems.BinaryProblems import *
from utils.Utils import run_experiments


def main():
    os.chdir('..')
    num_exec = 30
    num_particles = 100
    num_iter = 500
    smp = 5
    cdc = 0.8
    pmo = 0.5
    w = 1
    mr = 0.2
    c1 = 2.0

    out_dir = "results/"
    funcs = [ZeroOneKnapsack]
    dims = [10]

    for func in funcs:
        for d in dims:
            for run in range(num_exec):
                problem = func(d)
                output_dir = out_dir + "{}/{}/Dim_{}/Exec_{}/".format("BCSO", problem.name, problem.dim, run + 1)

                if not os.path.isdir(output_dir):
                    os.makedirs(output_dir)

                opt = BCSO(problem, swarm_size=num_particles, n_iter=num_iter, w=w, c1=c1, smp=smp, cdc=cdc, pmo=pmo,
                           mr=mr, output_dir=output_dir)
                run_experiments(opt, run, output_dir)


if __name__ == '__main__':
    main()
