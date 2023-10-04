import copy

import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

from networks.PortraitDivergence import *
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import collections, re
import seaborn as sns
from numpy import trapz


def gen_destruction_matrix(self, curr_iter):
    filter_range = np.linspace(0.0, 1.0, num=100)
    tw_range = np.linspace(0.0, curr_iter + 1, num=100)
    destruction = np.zeros((len(filter_range), len(tw_range)))

    for tw in range(len(tw_range)):
        g = self.__apply_time_window(curr_iter, tw_range[tw])
        destruction[:, tw] = (calculate_destruction_pace(g, filter_range))

    return destruction


def calculate_destruction_pace(g, filter_range):
    connected_comp = []
    temp_graph = g.to_undirected()
    edges = sorted(list(temp_graph.edges.data()), key=lambda tup: tup[2]['weight'])
    edges = [[x[0], x[1], x[2]['weight'] / edges[-1][2]['weight']] for x in edges]
    for f in filter_range:
        remove_list = [x for x in edges if x[2] <= f]
        edges = [x for x in edges if x[2] > f]
        for n1, n2, w in remove_list:
            temp_graph.remove_edge(n1, n2)
        connected_comp.append(nx.number_connected_components(temp_graph))
    return connected_comp


def array_split(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out


def generate_plots(com_graph, pos, out_loc, tw, name):
    com_graph = nx.convert_node_labels_to_integers(com_graph)
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(26, 14))

    gen_centrality_plot(com_graph, pos, fig, axes[0][0])
    gen_interaction_net_plot(com_graph, axes[0][1])
    gen_nodes_strength_plot(com_graph, axes[0][2])

    # TODO: Validate the code bellow
    # destruction = self.__gen_destruction_matrix(curr_iter)
    # gen_destruction_plot(destruction, curr_iter, fig, axes[1][0])
    # gen_destruction_curve_plot(destruction, curr_iter, self.population_size, axes[1][1])

    # if fitness:
    #     generate_fitness_plot(fitness, curr_iter, axes[1][2])

    fig.savefig(out_loc + "/tw_{}_{}_networks".format(tw, name), dpi=fig.dpi)
    plt.close('all')


def gen_centrality_plot(g, pos, fig, ax):
    ax.set_title("Degree Centrality")
    if g.is_directed():
        degree = np.array([float(i) for i in dict(g.in_degree).values()])
    else:
        degree = np.array([float(i) for i in dict(g.degree).values()])
    degree = degree / degree.max()
    measures = nx.degree_centrality(g)

    nodes = nx.draw_networkx_nodes(g, pos, node_size=[100 * v for v in degree], cmap=plt.cm.Spectral_r,
                                   node_color=list(measures.values()), nodelist=list(measures.keys()), ax=ax)

    nodes.set_norm(mcolors.SymLogNorm(linthresh=0.01, linscale=1))
    labels = nx.draw_networkx_labels(g, pos, ax=ax)
    nx.draw_networkx_edges(g, pos, label=labels, arrows=False, alpha=0.1, ax=ax)
    ax.set_axis_off()
    try:
        fig.colorbar(nodes, ax=ax)
    except:
        pass


def gen_interaction_net_plot(g, ax):
    ax.set_title("Interaction Network")
    matrix_ = nx.to_numpy_array(g)
    matrix_ = [np.array(r) for r in matrix_[:, np.argsort(matrix_.sum(axis=0))]]
    sns.heatmap(matrix_, xticklabels=False, yticklabels=False, cmap=plt.cm.Spectral_r, ax=ax)
    ax.invert_xaxis()


def gen_nodes_strength_plot(g, ax):
    ax.set_title("Strength of the Nodes")
    if g.is_directed():
        degree = pd.Series(list(dict(g.in_degree).values()), name="Node strength")
    else:
        degree = pd.Series(list(dict(g.degree).values()), name="Node strength")
    sns.distplot(degree, ax=ax)


# TODO: Verify method implementation
def gen_destruction_plot(destruction, curr_iter, fig, ax):
    ax.set_title("Destruction Pace")
    ax.set(xlabel="Time Window", ylabel="Filter (%)")
    c = ax.pcolorfast(destruction, cmap=plt.cm.Spectral_r)
    fig.colorbar(c, ax=ax)
    # plt.xticks(np.linspace(1, curr_iter + 1, num=10), ax=ax)
    # plt.yticks(np.linspace(0.0, 1.0, num=10), ax=ax)


# TODO: Verify method implementation
def gen_destruction_curve_plot(destruction, curr_iter, pop_size, ax):
    ax.set_title("Destruction Curve".format(curr_iter))

    ax.plot(destruction, linewidth=2)
    ax.set(xlabel="Filter (%)", ylabel="Number of Components")
    ax.axes.get_xaxis().set_visible(True)
    ax.axes.get_yaxis().set_visible(True)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=20)
    plt.setp(ax.yaxis.get_majorticklabels(), rotation=20)

    area = trapz(destruction, dx=5) / 10000
    cd = 1 - (1 / (pop_size * len(area))) * np.sum(area)
    # nx.draw(temp_g, cmap=plt.cm.Blues)


def generate_fitness_plot(fitness, curr_iter, ax):
    ax.set_title("Fitness Evolution - Best: {:.2E}".format(fitness[curr_iter - 1]))
    ax.plot(fitness[:curr_iter], color='red', linewidth=2)
    ax.set(xlabel="Iterations", ylabel="Fitness")
    ax.axes.get_xaxis().set_visible(True)
    ax.axes.get_yaxis().set_visible(True)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=20)
    plt.setp(ax.yaxis.get_majorticklabels(), rotation=20)


def join_networks(g, h):
    for n1, n2, temp_w in h.edges(data='weight'):
        w = g.get_edge_data(n1, n2)
        if type(w) is 'NoneType':
            g.add_edge(n1, n2, weight=(temp_w + w['weight']))
        else:
            g.add_edge(n1, n2, weight=temp_w)
    return g


def get_time_windows(a, execs, d, func, it, output_dir, tw, pop_size, model=""):
    name = output_dir + "/{}_Dim_{}_Exec_{}/{}/inet_graph.csv"

    graphs_execs_dict = {}
    net_poins = array_split(range(1, it + 1), int(it / tw))

    def __aux_tw(matrices_list):
        graphs_arr = [nx.from_numpy_matrix(np.sum(matrices_list[its[0]:its[-1], ], axis=0).reshape(pop_size, pop_size),
                                           create_using=nx.DiGraph()) for its in net_poins]
        return dict(zip(range(1, len(graphs_arr) + 1), graphs_arr))

    temp = np.loadtxt(name.format(a, d, str(1) + model, func))
    matrices_execs_cum = copy.deepcopy(temp)
    temp = __aux_tw(temp)[int(it / tw)]
    graphs_execs_dict[1] = temp

    for ex in range(2, execs + 1):
        temp = np.loadtxt(name.format(a, d, str(ex) + model, func))
        matrices_execs_cum += copy.deepcopy(temp)
        temp = __aux_tw(temp)[int(it / tw)]
        graphs_execs_dict[ex] = temp
    return graphs_execs_dict, __aux_tw(matrices_execs_cum)


def generate_networks(execs):
    gen_models = {"RND": nx.gnm_random_graph, "REG": nx.random_regular_graph, "WS": nx.watts_strogatz_graph,
                  "NWS": nx.newman_watts_strogatz_graph, "ER": nx.erdos_renyi_graph,
                  "BA": nx.barabasi_albert_graph}
    dicts = {}
    idx = 0
    for name, gen in gen_models.items():
        _graphs = {}
        g = None
        for x in range(execs):
            if name in ["WS", "NWS"]:
                h = gen(n=100, k=50, p=0.25)
            elif name in ["REG"]:
                h = gen(n=100, d=5)
            elif name in ["ER"]:
                h = gen(n=100, p=0.25)
            elif name in ["BA", "RND"]:
                h = gen(n=100, m=50)
            _graphs[x] = copy.deepcopy(h)
        dicts[name] = copy.deepcopy(_graphs)
        idx += 1
    return dicts


def plot_matrix(nets_a, nets_b, tw, ax_title, out_loc, plot_type):
    out_f = "/{}_pd_difference_results.txt"
    alg_weighted = []
    res = []
    l1 = []
    l2 = []
    f_handle = open(out_loc + out_f.format(nets_a[0]), 'w+')
    f_handle.close()
    for e1, g in sorted(nets_a[1].items(), reverse=True):
        line_w = []
        l1.append(e1)
        f_handle = open(out_loc + out_f.format(nets_a[0]), 'a+')
        for e2, h in sorted(nets_b[1].items(), reverse=False):
            l2.append(e2)
            w_djs, w_bg, w_bh = portrait_divergence_weighted(g, h, bins=1)
            res.append(w_djs)
            line_w.append(w_djs)
            f_handle.write("{}\n".format(w_djs))
        f_handle.close()
        alg_weighted.append(line_w)

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    xticks = sorted(range(1, len(nets_a[1]) + 1), reverse=True)
    yticks = sorted(range(1, len(nets_b[1]) + 1), reverse=False)
    sns.heatmap(alg_weighted, xticklabels=xticks, yticklabels=yticks, cmap=plt.cm.Spectral_r, annot=False, ax=ax,
                vmin=0, vmax=1)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=90)
    plt.setp(ax.yaxis.get_majorticklabels(), rotation=0)

    plt.xlabel(nets_a[0] + " - " + ax_title, fontsize=18, labelpad=18)
    plt.ylabel(nets_b[0] + " - " + ax_title, fontsize=18, labelpad=18)
    fig.savefig(out_loc + "/tw_{}_{}_{}_{}".format(tw, nets_a[0], nets_b[0], plot_type), dpi=fig.dpi)
    plt.show()


def gen_diff_plot(a, b, tw, out_loc):
    out_f = "/{}.fitness_difference_results.txt"
    natural_s = lambda k, v: [k, int(v)]
    diff_tab = []

    f_handle = open(out_loc + out_f.format(list(a.keys())[0].split("_")[0]), 'w+')
    f_handle.close()
    for a1, v1 in sorted(a.items(), key=lambda t: natural_s(*re.match(r'([a-zA-Z]+_+)+(\d+)', t[0]).groups()),
                         reverse=True):
        f_handle = open(out_loc + out_f.format(a1.split("_")[0]), 'a+')
        line = []
        for a2, v2 in sorted(b.items(), key=lambda t: natural_s(*re.match(r'([a-zA-Z]+_+)+(\d+)', t[0]).groups()),
                             reverse=False):
            f_handle.write("{} vs {}: {}\n".format(a1, a2, (v1 - v2)))
            line.append(v1 - v2)
        diff_tab.append(line)
        f_handle.close()

    diff_tab = np.asmatrix(diff_tab)
    diff_tab = (diff_tab - diff_tab.mean()) / diff_tab.std()

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    xticks = sorted(range(1, len(diff_tab) + 1), reverse=True)
    yticks = sorted(range(1, len(diff_tab) + 1), reverse=False)
    name1 = list(a.keys())[0].split("_")[0]
    name2 = list(b.keys())[0].split("_")[0]
    sns.heatmap(diff_tab, cmap=plt.cm.Spectral_r, annot=False, ax=ax, xticklabels=xticks, yticklabels=yticks)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=90)
    plt.setp(ax.yaxis.get_majorticklabels(), rotation=0)
    plt.xlabel(name1 + " - Time Windows", fontsize=18, labelpad=18)
    plt.ylabel(name2 + " - Time Windows", fontsize=18, labelpad=18)
    fig.savefig(out_loc + "/tw_{}_{}_{}_fitness".format(tw, name1, name2), dpi=fig.dpi)


def load_alg_networks(alg, exs, dim, func, it, tw, sim_dir, pop_size, out_loc):
    pos = nx.spring_layout(nx.complete_graph(pop_size), iterations=100)
    dicts_execs_list = {}
    dicts_execs_combined = {}

    for a in alg:
        print("Loading {}".format(a))
        graphs_execs_dict, graphs_execs_combined = get_time_windows(a, exs, dim, func, it, sim_dir, tw, pop_size)
        generate_plots(graphs_execs_combined[int(it / tw)], pos, out_loc, tw, a)
        dicts_execs_list[a] = graphs_execs_dict
        dicts_execs_combined[a] = graphs_execs_combined
    return dicts_execs_list, dicts_execs_combined
