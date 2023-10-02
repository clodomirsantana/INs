import os
from itertools import cycle

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats

matplotlib.rcParams.update({'font.size': 14})


def create_dir(path):
    directory = os.path.dirname(path)
    try:
        os.stat(directory)
    except Exception:
        os.mkdir(directory)


def swarm_evolution(swarm):
    xar = []
    yar = []
    s = []

    for ind in swarm:
        xar.append(ind.pos[0])
        yar.append(ind.pos[1])
        s.append(ind.weight)

    plt.ion()
    plt.clf()
    plt.scatter(xar, yar, s=s)
    plt.ylim([-100, 100])
    plt.xlim([-100, 100])
    plt.pause(0.05)


def generate_box_plots(files_loc, functions, algorithms, dimensions, num_iter, per_eval, grouped=True, time=False):
    plt.figure(figsize=(6, 4))
    title = "{} function {} dimensions"
    if per_eval:
        file_ = "_cost_eval.txt"
        file_exec_ = "_exec_time_eval.txt"
    else:
        file_ = "_cost_iter.txt"
        file_exec_ = "_exec_time_iter.txt"

    for idx in range(len(functions)):
        for dim in range(len(dimensions)):
            if grouped:
                plt.subplot(np.ceil(len(dimensions) / 3.0), 3, dim + 1)
            else:
                plt.figure()
            box = []
            for alg in algorithms:
                file_name = os.path.join(files_loc, alg, functions[idx],
                                         alg + "_" + functions[idx] + "_" + str(dimensions[dim]) + file_)
                f = open(file_name, 'r')
                execs = []
                for line in f:
                    data = line.split()
                    data = data[:num_iter]

                    data = [data[i] for i in range(num_iter)]
                    data = np.asarray(data, dtype=np.float)
                    execs.append(data[num_iter - 1])

                box.append(execs)

            plt.boxplot(box)
            # plt.title(title.format(functions[idx], dimensions[dim]))
            plt.xlabel('Heuristic')
            plt.ylabel('Fitness')
            plt.xticks(range(1, len(algorithms) + 1), algorithms)
            plt.yticks(rotation=45)
            plt.xticks(rotation=-45)
            if not grouped:
                plt.tight_layout()

            output_name_w = "_box_plots"
            plt.savefig(os.path.join(files_loc, functions[idx] + "_" + str(dimensions[dim]) + output_name_w))

            if time:
                for dim in range(len(dimensions)):
                    if grouped:
                        plt.subplot(np.ceil(len(dimensions) / 3.0), 3, dim + 1)
                    else:
                        plt.figure()
                    box_exec = []
                    for alg in algorithms:
                        file_exec_name = os.path.join(files_loc, alg, functions[idx],
                                                      alg + "_" + functions[idx] + "_" + str(
                                                          dimensions[dim]) + file_exec_)
                        f_exec = open(file_exec_name, 'r')
                        execs_time = []

                        for line in f_exec:
                            data = line.split()
                            data = data[:num_iter]

                            data = [data[i] for i in range(num_iter)]
                            data = np.asarray(data, dtype=np.float)
                            execs_time.append(data[num_iter - 1])

                        box_exec.append(execs_time)

                    plt.boxplot(box_exec)
                    # plt.title(title.format(functions[idx], dimensions[dim]))
                    plt.xlabel('Heuristic')
                    plt.ylabel('Execution Time')
                    plt.xticks(range(1, len(algorithms) + 1), algorithms)
                    plt.yticks(rotation=45)
                    plt.xticks(rotation=-45)
                    if not grouped:
                        plt.tight_layout()

                    output_name_exec_w = "_exec_time_box_plots"
                    plt.savefig(
                        os.path.join(files_loc, functions[idx] + "_" + str(dimensions[dim]) + output_name_exec_w))


def generate_lines_plot(files_loc, functions, algorithms, dimensions, num_iter, per_eval, grouped, n_exec=30):
    plt.figure(figsize=(6, 4))

    title = "{} function {} dimensions"
    if per_eval:
        file_ = "_cost_eval.txt"
        x_label = 'Evaluations'
    else:
        file_ = "_cost_iter.txt"
        x_label = 'Iterations'

    for idx in range(len(functions)):

        for dim in range(len(dimensions)):
            markers = ['o', '^', 'v', '<', '>', 'P', '1', '2', '3', '4', '8', 's', 'p', '.', 'h', 'H', '+', '*']
            markerscycler = cycle(markers)
            lines_ = ["-", "--", "-.", ":"]
            linecycler = cycle(lines_)
            if grouped is True:
                plt.subplot(np.ceil(len(dimensions) / 3.0), 3, dim + 1)
            else:
                plt.figure()

            for alg in algorithms:
                file_name = os.path.join(files_loc, alg, functions[idx],
                                         alg + "_" + functions[idx] + "_" + str(dimensions[dim]) + file_)
                f = open(file_name, 'r')
                count = 0
                lines = []
                for line in f:
                    data = line.split()
                    data = data[:num_iter]
                    data = [0 if float(x) == float("-inf") else x for x in data]
                    if count == 0:
                        lines = np.asarray(data, dtype=np.float)
                    elif count < n_exec:
                        lines = np.asarray(data, dtype=np.float) + lines
                    count += 1
                    if count == n_exec:
                        break
                lines = lines / count
                markers_on = range(0, num_iter, (num_iter / 10))
                plt.plot([lines[i] for i in range(30, num_iter)], linestyle=next(linecycler),
                         marker=next(markerscycler), markevery=markers_on)
                # plt.yscale('log')

            # plt.title(title.format(functions[idx], dimensions[dim]))
            plt.legend(algorithms, loc='lower right', ncol=3)
            plt.xlabel(x_label)
            plt.ylabel('Fitness')
            plt.yticks(rotation=45)
            plt.xticks(rotation=-45)
            if not grouped:
                plt.tight_layout()

            output_name_w = "_lines_plot"
            # plt.show()
            plt.savefig(os.path.join(files_loc, functions[idx] + "_" + str(dimensions[dim]) + output_name_w))


def generate_stats_test_results(files_loc, functions, algorithms, dimensions, num_iter, per_eval, is_fs, time=False):
    if per_eval:
        file_ = "_cost_eval.txt"
        file_acc_test = "_acc_test_eval.txt"
        file_acc_train = "_acc_train_eval.txt"
        file_fs = "_features_eval.txt"
        file_time = "_exec_time_eval.txt"
        output_name_w = "_wicoxon_test_results_eval.csv"
    else:
        file_ = "_cost_iter.txt"
        file_acc_test = "_acc_test_iter.txt"
        file_acc_train = "_acc_train_iter.txt"
        file_fs = "_features_iter.txt"
        file_time = "_exec_time_iter.txt"
        output_name_w = "_wicoxon_test_results_iter.csv"

    for idx in range(len(functions)):
        print("======================= FUNC: {} =======================".format(functions[idx]))
        for dim in range(len(dimensions)):
            fitness_line = '& Fitness'
            acc_train_line = '& Accuracy'
            fit_wilcox_line = '& Wilcox'

            sf_line = '& \#SF'
            acc_test_line = '& Accuracy'
            sf_wilcox_line = '& Wilcox'

            exec_time_line = '& Exec Time'
            print("======================= DIM: {} =======================".format(dim))
            populations = []
            populations_acc_train = []
            populations_acc_test = []
            populations_fs = []
            populations_ex = []
            for alg in algorithms:

                if is_fs:
                    file_name_acc_train = os.path.join(files_loc, alg, functions[idx],
                                                       alg + "_" + functions[idx] + "_" + str(
                                                           dimensions[dim]) + file_acc_train)
                    file_name_acc_test = os.path.join(files_loc, alg, functions[idx],
                                                      alg + "_" + functions[idx] + "_" + str(
                                                          dimensions[dim]) + file_acc_test)
                    file_name_fs = os.path.join(files_loc, alg, functions[idx],
                                                alg + "_" + functions[idx] + "_" + str(dimensions[dim]) + file_fs)

                    f_acc_train = open(file_name_acc_train, 'r')
                    f_acc_test = open(file_name_acc_test, 'r')
                    f_fs = open(file_name_fs, 'r')

                    execs_acc_test = []
                    execs_acc_train = []
                    execs_fs = []

                    for line in f_acc_train:
                        data_acc_train = line.split()
                        data_acc_train = np.asarray(data_acc_train, dtype=np.float)
                        execs_acc_train.append(data_acc_train[num_iter - 1])
                    populations_acc_train.append(execs_acc_train)

                    for line in f_acc_test:
                        data_acc_test = line.split()
                        data_acc_test = np.asarray(data_acc_test, dtype=np.float)
                        execs_acc_test.append(data_acc_test[num_iter - 1])
                    populations_acc_test.append(execs_acc_test)

                    for line in f_fs:
                        data_fs = line.split()
                        data_fs = np.asarray(data_fs, dtype=np.float)
                        execs_fs.append(data_fs[num_iter - 1])
                    populations_fs.append(execs_fs)

                    acc_train_line = acc_train_line + " & {:.3f} $\\pm${:.3f}".format(
                        round(np.mean(execs_acc_train), 3),
                        round(np.std(execs_acc_train), 3))

                    sf_line = sf_line + " & {:.3f} $\\pm${:.3f}".format(round(np.mean(execs_fs), 3),
                                                                        round(np.std(execs_fs), 3))

                    acc_test_line = acc_test_line + " & {:.3f} $\\pm${:.3f}".format(round(np.mean(execs_acc_test), 3),
                                                                                    round(np.std(execs_acc_test), 3))

                file_name = os.path.join(files_loc, alg, functions[idx],
                                         alg + "_" + functions[idx] + "_" + str(dimensions[dim]) + file_)
                file_name_exec = os.path.join(files_loc, alg, functions[idx],
                                              alg + "_" + functions[idx] + "_" + str(dimensions[dim]) + file_time)

                f = open(file_name, 'r')

                execs = []
                execs_ex = []

                if time:
                    f_ex = open(file_name_exec, 'r')
                    for line in f_ex:
                        data_ex = line.split()
                        data_ex = np.asarray(data_ex, dtype=np.float)
                        execs_ex.append(data_ex[num_iter - 1])
                    populations_ex.append(execs_ex)
                    exec_time_line = exec_time_line + " & {:.3f} $\\pm${:.3f}".format(round(np.mean(execs_ex), 3),
                                                                                      round(np.std(execs_ex), 3))

                for line in f:
                    data = line.split()
                    data = np.asarray(data, dtype=np.float)
                    execs.append(data[num_iter - 1])
                populations.append(execs)

                if is_fs:
                    fitness_line = fitness_line + " & {:.3f} $\\pm${:.3f}".format(round(np.mean(execs), 3),
                                                                                  round(np.std(execs), 3))
                if not is_fs:
                    fitness_line = fitness_line + " & {:.3f}".format(round(np.mean(execs), 3))
                    acc_train_line = acc_train_line + " & $\\pm${:.3f}".format(round(np.std(execs), 3))

            alg1 = 0
            matrix = []
            for alg2 in range(len(algorithms)):
                test = scipy.stats.ranksums(populations[alg1], populations[alg2])
                if test[1] < 0.05:
                    if test[0] > 0:
                        fit_wilcox_line = fit_wilcox_line + " & $\\blacktriangle$"
                        result = "better than"
                    else:
                        fit_wilcox_line = fit_wilcox_line + " & $\\blacktriangledown$"
                        result = "worst than"
                else:
                    result = "equals to"
                    fit_wilcox_line = fit_wilcox_line + " & --"
                matrix.append(
                    ["Algorithm {} {} {}".format(algorithms[alg1], result, algorithms[alg2])])

            if is_fs:
                for alg2 in range(len(algorithms)):
                    test = scipy.stats.ranksums(populations_fs[alg1], populations_fs[alg2])
                    if test[1] < 0.05:
                        if test[0] < 0:
                            sf_wilcox_line = sf_wilcox_line + " & $\\blacktriangle$"
                            result = "better than"
                        else:
                            sf_wilcox_line = sf_wilcox_line + " & $\\blacktriangledown$"
                            result = "worst than"
                    else:
                        sf_wilcox_line = sf_wilcox_line + " & --"
                        result = "equals to"
                    matrix.append(
                        ["Algorithm {} {} {}".format(algorithms[alg1], result, algorithms[alg2])])

            df = pd.DataFrame(matrix)
            new_header = df.iloc[0]
            df = df[1:]
            df = df.rename(columns=new_header)
            df.to_csv(os.path.join(files_loc, functions[idx] + "_" + algorithms[alg1] + "_" + str(
                dimensions[dim]) + output_name_w), index=False)

            print("============= Fitness Values Table =============")
            print(fitness_line + " \\\\")
            print(acc_train_line + " \\\\")
            print(fit_wilcox_line + " \\\\")

            if is_fs:
                print("============= Selected Features Table =============")
                print(sf_line + " \\\\")
                print(acc_test_line + " \\\\")
                print(sf_wilcox_line + " \\\\")

            if time:
                print("============= Execution Time =============")
                print(exec_time_line + " \\\\")

        if time:
            generate_time_results(files_loc, functions, algorithms, dimensions, num_iter, per_eval)


def generate_time_results(files_loc, functions, algorithms, dimensions, num_iter, per_eval):
    if per_eval:
        file_time = "_exec_time_eval.txt"
    else:
        file_time = "_exec_time_iter.txt"

    for idx in range(len(functions)):
        res_mean = []
        res_std = []
        for alg in algorithms:
            populations_ex = []
            for dim in range(len(dimensions)):
                file_name_exec = os.path.join(files_loc, alg, functions[idx],
                                              alg + "_" + functions[idx] + "_" + str(dimensions[dim]) + file_time)

                f_ex = open(file_name_exec, 'r')
                execs_ex = []

                for line in f_ex:
                    data_ex = line.split()
                    data_ex = np.asarray(data_ex, dtype=np.float)
                    execs_ex.append(data_ex[num_iter - 1])
                populations_ex.append(execs_ex)
            low = min([len(x) for x in populations_ex])
            populations_ex = [ka[:low] for ka in populations_ex]
            res_std.append(np.std(populations_ex, axis=1))
            res_mean.append(np.mean(populations_ex, axis=1))

        res_std = pd.DataFrame(res_std)
        res_std.columns = dimensions
        res_std = res_std.transpose()
        res_std.columns = algorithms
        res_mean = pd.DataFrame(res_mean)
        res_mean.columns = dimensions
        res_mean = res_mean.transpose()
        res_mean.columns = algorithms

        res_mean.plot.bar(width=0.8)
        plt.xlabel("Dimensions")
        plt.ylabel('Time (Seconds)')
        plt.yticks(rotation=45)
        plt.xticks(rotation=-45)
        # plt.tight_layout()

        output_name_exec_w = "_exec_time_bar_plots"
        plt.savefig(os.path.join(files_loc, functions[idx] + output_name_exec_w))
        plt.show()


def run_experiments(opt, run, save_dir):
    console_out = "Alg: {} Problem: {} Dim: {} Exec: {} Best Cost: {:.2E}"
    init_message = "Execution Started - Algorithm: {} Problem: {} Dim:  {}"

    print(init_message.format(opt.name, opt.objective_function.name, opt.dim))

    opt.optimize()
    print(console_out.format(opt.name, opt.objective_function.name, opt.dim, run + 1, opt.best_agent.cost))

    save_simulation_data(opt, save_dir, run)


def save_simulation_data(opt, save_dir, exc):
    files = ["{}/cost_iter.txt".format(save_dir),
             "{}/swarm_cost_iter.txt".format(save_dir),
             "{}/curr_best_cost_iter.txt".format(save_dir),
             "{}/curr_worst_cost_iter.txt".format(save_dir),
             "{}/pos_diff_mean_iter.txt".format(save_dir),
             "{}/exec_time_iter.txt".format(save_dir)]

    values = [opt.optimum_cost_tracking_iter, opt.swarm_cost_tracking_iter, opt.curr_best_cost_tracking_iter,
              opt.curr_worst_cost_tracking_iter, opt.pos_diff_mean_iter, opt.execution_time_tracking_iter]

    for idx in range(len(files)):
        if exc > 1:
            f_handle = open(files[idx], 'a')
        else:
            f_handle = open(files[idx], 'w+')

        temp_values = np.asmatrix(values[idx])
        np.savetxt(f_handle, temp_values, fmt='%.4e')
        f_handle.close()
