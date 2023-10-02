from utils.Utils import *


def plot_generator(files_loc, functions, algorithms, dimensions, num_iter, num_eval=None, lines_plot=True,
                   box_plot=True, wilcox=True, grouped=True, is_fs=False, time=False):
    if lines_plot:
        print("Generating Lines plot")
        if num_eval:
            generate_lines_plot(files_loc=files_loc, functions=functions, algorithms=algorithms,
                                dimensions=dimensions, num_iter=num_eval, per_eval=True, grouped=grouped, n_exec=30)
        if num_iter:
            generate_lines_plot(files_loc=files_loc, functions=functions, algorithms=algorithms, num_iter=num_iter,
                                dimensions=dimensions, per_eval=False, grouped=grouped, n_exec=30)

    if box_plot:
        print("Generating Box plot")
        if num_eval:
            generate_box_plots(files_loc=files_loc, functions=functions, algorithms=algorithms,
                               dimensions=dimensions, num_iter=num_eval, per_eval=True, time=time, grouped=grouped)
        if num_iter:
            generate_box_plots(files_loc=files_loc, functions=functions, algorithms=algorithms,
                               dimensions=dimensions, num_iter=num_iter, per_eval=False, time=time, grouped=grouped)

    if any([wilcox]):
        print("Applying Tests")
        if num_eval:
            generate_stats_test_results(files_loc=files_loc, functions=functions, algorithms=algorithms, time=time,
                                        dimensions=dimensions, num_iter=num_eval, per_eval=True, is_fs=is_fs)
        if num_iter:
            generate_stats_test_results(files_loc=files_loc, functions=functions, algorithms=algorithms, time=time,
                                        dimensions=dimensions, num_iter=num_iter, per_eval=False, is_fs=is_fs)


def main():
    os.chdir('../..')
    os.getcwd()
    alg_type = "discrete"
    files_loc = os.getcwd() + "/results/{}".format(alg_type)
    is_fs = False
    time = True
    dimensions = [50]
    funcs = ["Sphere"]

    algorithms = ["ABC", "GPSO", "CSO"]
    num_iter = 500

    plot_generator(files_loc=files_loc, functions=funcs, dimensions=dimensions, algorithms=algorithms, is_fs=is_fs,
                   num_iter=num_iter, num_eval=num_eval, lines_plot=True, box_plot=False, wilcox=True, grouped=False,
                   time=time)


if __name__ == '__main__':
    main()
