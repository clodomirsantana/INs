from algorithms.GPSO import GPSO
from problems.ObjectiveFunction import *
from utils.SearchSpaceInitializer import UniformSSInitializer
from utils.Utils import *


def main():
    os.chdir('..')
    num_exec = 30
    num_particles = 100
    num_iter = 500
    lb_w = 0.4
    up_w = 0.9
    c1 = (0.72984 * 2.05)
    c2 = (0.72984 * 2.05)

    search_space_initializer = UniformSSInitializer()
    out_dir = "results/"
    funcs = [Sphere]
    dims = [50]

    for func in funcs:
        for d in dims:
            for run in range(num_exec):
                problem = func(d)
                output_dir = out_dir + "{}/{}/Dim_{}/Exec_{}/".format("GPSO", problem.name, problem.dim, run + 1)

                if not os.path.isdir(output_dir):
                    os.makedirs(output_dir)

                opt = GPSO(problem, search_space_initializer, swarm_size=num_particles, n_iter=num_iter, lb_w=lb_w,
                           up_w=up_w, c1=c1, c2=c2, v_max=10, output_dir=output_dir)
                run_experiments(opt, run, output_dir)


if __name__ == '__main__':
    main()
