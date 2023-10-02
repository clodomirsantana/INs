from algorithms.CSO import CSO
from problems.ObjectiveFunction import *
import os

from utils.SearchSpaceInitializer import *
from utils.Utils import run_experiments


def main():
    os.chdir('..')
    num_exec = 30
    swarm_size = 100
    num_iter = 500

    up_w = 0.9
    lw_w = 0.4
    smp = 5
    srd = 0.2
    cdc = 0.8
    mr = 0.2
    c1 = 2.0
    max_vel = 10

    search_space_initializer = UniformSSInitializer()
    out_dir = "results/"
    funcs = [Sphere]
    dims = [50]

    for func in funcs:
        for d in dims:
            for run in range(num_exec):
                problem = func(d)
                output_dir = out_dir + "{}/{}/Dim_{}/Exec_{}/".format("CSO", problem.name, problem.dim, run + 1)

                if not os.path.isdir(output_dir):
                    os.makedirs(output_dir)

                opt = CSO(objective_function=problem, search_space_initializer=search_space_initializer,
                          swarm_size=swarm_size, n_iter=num_iter, c1=c1, srd=srd, cdc=cdc, mr=mr, smp=smp, up_w=up_w,
                          lw_w=lw_w, max_vel=max_vel, output_dir=output_dir)
                run_experiments(opt, run, output_dir)


if __name__ == '__main__':
    main()
