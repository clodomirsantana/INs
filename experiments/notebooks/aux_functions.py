import math
import pickle
import toytree
import toyplot
import sys, os
import warnings
import toyplot.svg
import pandas as pd
import networkx as nx
import multiprocessing
import scipy.stats as sc
import matplotlib as mpl
from tqdm.auto import tqdm
from scipy.stats import entropy
import matplotlib.style as style
from scipy.cluster import hierarchy
from joblib import Parallel, delayed
from matplotlib.colors import LogNorm
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import ks_2samp, ranksums, wilcoxon, shapiro
from scipy.stats import pearsonr, mannwhitneyu, kendalltau, spearmanr

rc_defaults = dict(mpl.rcParams)
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.getcwd(), '../..'))
from networks.Utils import *


def entropy1(dist):
    """
    Returns the entropy of `dist` in bits (base-2).
    """
    dist = np.asarray(dist)
    ent = np.nansum(dist * np.log2(1 / dist))
    return ent


def estimate_shannon_entropy(dna_sequence):
    bases = collections.Counter([tmp_base for tmp_base in dna_sequence])
    # define distribution
    dist = [x / sum(bases.values()) for x in bases.values()]

    # use scipy to calculate entropy
    entropy_value = entropy(dist, base=2)

    return entropy_value


def centrality_distribution(G):
    """
    Returns a centrality distribution.
    Each normalized centrality is divided by the sum of the normalized
    centralities. Note, this assumes the graph is simple.
    """
    centrality = list(nx.degree_centrality(G).values())
    centrality = np.asarray(centrality)
    centrality /= centrality.sum()
    return centrality


def get_time_window(a, execs, d, func, it, output_dir, tw, pop_size, model=''):
    net_poins = array_split(range(it), int(it / tw))
    graphs_execs_dict = {}

    def _aux_get_graphs(a, d, ex_, func, i):
        network_file = output_dir + '/{}/{}/Dim_{}/Exec_{}/INs/inet_graph_iter{}.gml'
        G = nx.read_gml(network_file.format(a, func, d, ex_, i))
        G.remove_nodes_from(list(nx.isolates(G)))
        return (i, G)

    def _aux_combine_tws(its, graphs_):
        g_ = nx.DiGraph()
        for i in its:
            for n1, n2, w in graphs_[i].edges(data=True):
                w_ = g_.get_edge_data(n1, n2)
                try:
                    g_.add_edge(n1, n2, weight=float(w['weight']) + float(w_['weight']), count=w['count'] + 1)
                except:
                    g_.add_edge(n1, n2, weight=float(w['weight']), count=1)
                try:
                    g_.nodes[n1]['fit'] += float(graphs_[i].nodes[n1]['fit'])
                    g_.nodes[n1]['count'] += 1
                    g_.nodes[n1]['pos'] += graphs_[i].nodes[n1]['pos']
                    g_.nodes[n2]['fit'] += float(graphs_[i].nodes[n2]['fit'])
                    g_.nodes[n2]['count'] += 1
                    g_.nodes[n2]['pos'] += graphs_[i].nodes[n2]['pos']

                except:
                    g_.nodes[n1]['fit'] = float(graphs_[i].nodes[n1]['fit'])
                    g_.nodes[n1]['count'] = 1
                    g_.nodes[n1]['pos'] = graphs_[i].nodes[n1]['pos']
                    g_.nodes[n2]['fit'] = float(graphs_[i].nodes[n2]['fit'])
                    g_.nodes[n2]['count'] = 1
                    g_.nodes[n2]['pos'] = graphs_[i].nodes[n2]['pos']

        for n1, n2, w in g_.edges(data=True):
            w_ = g_.get_edge_data(n1, n2)
            g_.add_edge(n1, n2, weight=(float(w['weight']) / float(w_['count'])))

        for n in g_.nodes:
            g_.nodes[n]['fit'] = g_.nodes[n]['fit'] / g_.nodes[n]['count']
            g_.nodes[n]['pos'] = g_.nodes[n]['pos'] / g_.nodes[n]['count']

        return copy.deepcopy(g_)

    def aux_get_tw(ex, tw, it):
        graphs_, merged_tw = {}, []
        num_cores = multiprocessing.cpu_count()
        graphs_ = Parallel(n_jobs=1)(
            delayed(_aux_get_graphs)(a, d, str(ex) + model, func, i) for i in tqdm(range(it), leave=False))
        merged_tw = Parallel(n_jobs=1)(
            delayed(_aux_combine_tws)(its, dict(graphs_)) for its in tqdm(net_poins, leave=False))
        return (ex, dict(zip(range(1, int(it / tw) + 1), merged_tw)))

    num_cores = multiprocessing.cpu_count()
    graphs_execs_dict = Parallel(n_jobs=num_cores)(
        delayed(aux_get_tw)(ex, tw, it) for ex in tqdm(range(1, execs + 1), leave=False))
    return dict(graphs_execs_dict)


def get_time_windows(a, execs, d, func, it, output_dir, tw, pop_size, model=''):
    net_poins = array_split(range(it), int(it / tw))
    graphs_execs_dict = {}

    def _aux_get_graphs(a, d, exc, func, i):
        network_file = output_dir + '/{}/{}/Dim_{}/Exec_{}/INs/inet_graph_iter{}.gml'
        return (i, nx.read_gml(network_file.format(a, func, d, exc, i)))

    def _aux_combine_tws(its, graphs_):
        g_ = nx.DiGraph()
        for i in its:
            for n1, n2, w in graphs_[i].edges(data=True):
                w_ = g_.get_edge_data(n1, n2)
                try:
                    g_.add_edge(n1, n2, weight=float(w['weight']) + float(w_['weight']), count=w['count'] + 1)
                except:
                    g_.add_edge(n1, n2, weight=float(w['weight']), count=1)
                try:
                    g_.nodes[n1]['fit'] += float(graphs_[i].nodes[n1]['fit'])
                    g_.nodes[n1]['count'] += 1
                    g_.nodes[n1]['pos'] += graphs_[i].nodes[n1]['pos']
                    g_.nodes[n2]['fit'] += float(graphs_[i].nodes[n2]['fit'])
                    g_.nodes[n2]['count'] += 1
                    g_.nodes[n2]['pos'] += graphs_[i].nodes[n2]['pos']

                except:
                    g_.nodes[n1]['fit'] = float(graphs_[i].nodes[n1]['fit'])
                    g_.nodes[n1]['count'] = 1
                    g_.nodes[n1]['pos'] = graphs_[i].nodes[n1]['pos']
                    g_.nodes[n2]['fit'] = float(graphs_[i].nodes[n2]['fit'])
                    g_.nodes[n2]['count'] = 1
                    g_.nodes[n2]['pos'] = graphs_[i].nodes[n2]['pos']

        for n1, n2, w in g_.edges(data=True):
            w_ = g_.get_edge_data(n1, n2)
            g_.add_edge(n1, n2, weight=(0.0001 + float(w['weight']) / float(w_['count'])))

        for n in g_.nodes:
            g_.nodes[n]['fit'] = g_.nodes[n]['fit'] / g_.nodes[n]['count']

        return copy.deepcopy(g_)

    def aux_get_tw(ex, tw, it):
        graphs_, merged_tw = {}, []
        num_cores = multiprocessing.cpu_count()
        graphs_ = Parallel(n_jobs=1)(
            delayed(_aux_get_graphs)(a, d, str(ex) + model, func, i) for i in tqdm(range(it), leave=False))
        merged_tw = Parallel(n_jobs=1)(
            delayed(_aux_combine_tws)(its, dict(graphs_)) for its in tqdm(net_poins, leave=False))
        return (ex, dict(zip(range(1, int(it / tw) + 1), merged_tw)))

    num_cores = multiprocessing.cpu_count() - 1
    graphs_execs_dict = Parallel(n_jobs=num_cores)(
        delayed(aux_get_tw)(ex, tw, it) for ex in tqdm(range(1, execs + 1), leave=False))
    return dict(graphs_execs_dict)


def get_id_data(dicts_execs):
    res_id = {}
    for alg_name, data in tqdm(dicts_execs.items()):
        pds_weight_c, comps_weight_c = [], []
        pds_weight_w, comps_weight_w = [], []
        for exec_, networks in data.items():
            line_pds_weight_c, line_comps_weight_c = [], []
            line_pds_weight_w, line_comps_weight_w = [], []
            for _, net in networks.items():
                # ID values
                comp_weight_c = interaction_diversity(net)
                line_comps_weight_c.append(comp_weight_c)
            comps_weight_c.append(line_comps_weight_c)
        comps_weight_mean_c = np.mean(comps_weight_c, axis=0)
        comps_weight_std_c = np.std(comps_weight_c, axis=0)
        res_id[alg_name] = [comps_weight_std_c, comps_weight_std_c]
    return res_id


def plot_id_curves(res_id, show_std=False):
    num_alg = len(res_id)
    fig, axes = plt.subplots(nrows=(num_alg // 4), ncols=4, figsize=(12 * 4, 10 * (num_alg // 4)))
    idx = 0
    for alg_name, values in res_id.items():
        means = values[:len(values) // 2]
        stds = values[len(values) // 2:]
        for n, color, y in eczip(means, cmap='tab10', start=0):
            if show_std:
                std_u = y - stds[n]
                std_l = y + stds[n]
                norm_std = [max([np.abs(std_u[i]), np.abs(std_l[i])]) for i in range(len(y))]
                axes[idx // 4][idx % 4].plot(y / max(norm_std), color=color, linewidth=5.0)
                axes[idx // 4][idx % 4].fill_between(range(len(y)), std_u / max(norm_std), std_l / max(norm_std),
                                                     color=color, alpha=0.3)
            else:
                axes[idx // 4][idx % 4].plot(y / max(y), color=color, linewidth=8.0)

            axes[idx // 4][idx % 4].set_title(alg_name)

        idx += 1
    plt.legend(["ID"], fontsize=18, loc='center left', bbox_to_anchor=(1, 0.5))


def interaction_diversity(g, tw=50):
    g = g.to_undirected()
    pds_weight_c, comp_weight_c = [], []
    pds_weight_w, comp_weight_w = [], []
    perc_x = [0, .05, .1, .2, .4, .6, .75, 1]
    for f in perc_x:
        # removal by edge centrality and weight 
        g4 = copy.deepcopy(g)
        g5 = copy.deepcopy(g)

        d = nx.edge_betweenness_centrality(g4)
        max_c = max(d.values())
        edges_c = [[k[0], k[1], v / max_c] for k, v in sorted(d.items(), key=lambda item: item[1])]
        remove_list_c = [x for x in edges_c if x[2] <= f]

        edges_w = sorted(list(g5.edges.data()), key=lambda tup: tup[2]['weight'])
        max_w = max([v[2]['weight'] for v in edges_w]) + 0.0001
        edges_w = [[x[0], x[1], 1 - (x[2]['weight'] / max_w)] for x in edges_w]
        remove_list_w = [x for x in edges_w if x[2] <= f]

        for n1, n2, _ in remove_list_c:
            g4.remove_edge(n1, n2)

        for n1, n2, _ in remove_list_w:
            g5.remove_edge(n1, n2)

        # ID via components
        comp_weight_c.append(nx.number_connected_components(g4))

    area_comp_weight_c = np.trapz(y=comp_weight_c, x=perc_x)
    return (1 - area_comp_weight_c / len(g))


def load_alg_network(algorithms, pop_size, num_itr, num_excs, prob, prob_dim, tw_size, sim_dir):
    pos = nx.spring_layout(nx.complete_graph(pop_size), iterations=100)
    dicts_execs_list = {}

    for algorithm in tqdm(algorithms, leave=False):
        graphs_execs_dict = get_time_windows(algorithm, num_excs, prob_dim, prob, num_itr, sim_dir, tw_size, pop_size)
        dicts_execs_list[algorithm] = graphs_execs_dict
    return dicts_execs_list


def plot_matrix_exec(nets_a, nets_b, tw, out_loc):
    out_f = '/{}_pd_difference_results_exec.txt'
    res_mat = []

    f_handle = open(out_loc + out_f.format(nets_a[0]), 'w+')
    f_handle.close()
    print(tw)
    l1 = []
    for ex1 in sorted(range(1, len(nets_a[1]) + 1), reverse=True):
        l2 = []
        line_w = []
        l1.append(ex1)
        f_handle = open(out_loc + out_f.format(nets_a[0]), 'a+')
        for ex2 in sorted(range(1, len(nets_b[1]) + 1), reverse=False):
            l2.append(ex2)
            w_djs, w_bg, w_bh = portrait_divergence_weighted(nets_a[1][ex1][list(nets_a[1][ex1].keys())[-1]],
                                                             nets_b[1][ex2][list(nets_b[1][ex2].keys())[-1]])
            line_w.append(w_djs)
            f_handle.write('{}\n'.format(w_djs))
        f_handle.close()
        res_mat.append(line_w)

    res_mat = np.matrix(res_mat)
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    sns.heatmap(res_mat, yticklabels=l1, xticklabels=l2, cmap=plt.cm.Spectral_r, annot=False, ax=ax,
                vmin=0, vmax=1)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=90)
    plt.setp(ax.yaxis.get_majorticklabels(), rotation=0)

    plt.xlabel(str(nets_a[0]) + ' - ' + 'Execution', fontsize=18, labelpad=18)
    plt.ylabel(str(nets_b[0]) + ' - ' + 'Execution', fontsize=18, labelpad=18)
    fig.savefig(out_loc + '/ex_{}_{}_{}_{}'.format(tw, nets_a[0], nets_b[0], 'execution'), dpi=fig.dpi)
    plt.show()


def plot_matrix_tw(nets_a, nets_b, tw, out_loc):
    out_f = '/{}_{}_pd_difference_results_tw.txt'
    res_mat = []

    f_handle = open(out_loc + out_f.format(nets_a[0], nets_b[0]), 'w+')
    f_handle.close()
    print(nets_a[0], nets_b[0])
    for ex in range(1, len(nets_a[1]) + 1):
        alg_weighted = []
        l1 = []
        for e1, g in sorted(nets_a[1][ex].items(), reverse=True):
            l2 = []
            line_w = []
            l1.append(e1)
            f_handle = open(out_loc + out_f.format(nets_a[0], nets_b[0]), 'a+')
            for e2, h in sorted(nets_b[1][ex].items(), reverse=False):
                l2.append(e2)
                w_djs, w_bg, w_bh = portrait_divergence_weighted(g, h)
                line_w.append(w_djs)
                f_handle.write('{}\n'.format(w_djs))
            f_handle.close()
            alg_weighted.append(line_w)
        res_mat.append(alg_weighted)

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # Time Window Comparison plot
    mean_matrix = np.mean(res_mat, axis=0)

    sns.heatmap(mean_matrix, yticklabels=l1, xticklabels=l2, cmap=plt.cm.Spectral_r, annot=False, ax=ax,
                vmin=0, vmax=1)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=90)
    plt.setp(ax.yaxis.get_majorticklabels(), rotation=0)

    plt.xlabel(str(nets_a[0]) + ' - ' + 'Time Window', fontsize=18, labelpad=18)
    plt.ylabel(str(nets_b[0]) + ' - ' + 'Time Window', fontsize=18, labelpad=18)
    fig.savefig(out_loc + '/tw_{}_{}_{}_{}'.format(tw, nets_a[0], nets_b[0], 'time_windows'), dpi=fig.dpi)
    plt.show()


def get_tree_structure(linkage_, labels):
    def preorderTraversal(root, answer):
        if root.is_leaf():
            answer.append(labels[root.get_id()])
            return
        else:
            answer.append("(")
            preorderTraversal(root.get_left(), answer)
            answer.append(",")
            preorderTraversal(root.get_right(), answer)
            answer.append(")")

    rootnode, nodelist = hierarchy.to_tree(linkage_, rd=True)
    answer = []
    answer.append("(")
    preorderTraversal(rootnode, answer)
    answer.append(");")
    return ''.join(answer)


def plot_sample_network(alg_name, network):
    print(alg_name)
    network = nx.convert_node_labels_to_integers(network)
    d = centrality_distribution(network)
    labels = [np.around(network.nodes[n]['fit'], 2) for n in network.nodes]
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(26, 8))

    pos = nx.spring_layout(network)
    gen_centrality_plot(network, pos, fig, axes[0])
    gen_interaction_net_plot(network, axes[1])
    gen_nodes_strength_plot(network, axes[2])

    plt.show()


def load_data_per_tw(alg, num_execs, num_iter, tw, dim, func, output_dir):
    ftype = ['cost', 'swarm_cost', 'curr_best_cost', 'curr_worst_cost', 'pos_diff_mean']
    res_mean = []
    res_std = []

    for ft in ftype:
        lines = []
        fits_mean = []
        fits_std = []
        for ex in range(1, num_execs + 1):
            file_n = output_dir + f"/{alg}/{func}/Dim_{dim}/Exec_{ex}/{ft}_iter.txt"
            f = open(file_n, 'r')
            line = f.readlines()[-1]
            data = line.split()
            data = data[:num_iter]
            data = [0 if float(x) == float('-inf') else x for x in data]
            lines.append(np.asarray(data, dtype=np.float))

        lines_mean = np.mean(lines, axis=0)
        lines_std = np.std(lines, axis=0)
        net_poins = array_split(range(0, num_iter), int(num_iter / tw))
        for iters in net_poins:
            fit_mean = lines_mean[iters[0]]
            fit_std = lines_std[iters[0]]
            for itr in iters[1:]:
                fit_mean += lines_mean[itr]
                fit_std += lines_std[itr]
            fits_mean.append(fit_mean / len(iters))
            fits_std.append(fit_std / len(iters))

        res_mean.append(np.array(fits_mean))
        res_std.append(np.array(fits_std))

    res_mean.extend(res_std)
    return res_mean


def get_network_data(algs_funcs_dict):
    res_nets = {}

    for alg_name, data in algs_funcs_dict.items():
        execs_weight_mean, execs_weight_std = [], []
        node_degree_mean, node_degree_std = [], []
        node_fit_mean, node_fit_std = [], []

        # Executions loop
        for exec_, networks in data.items():
            line_weight_mean, line_weight_std = [], []
            line_degree_mean, line_degree_std = [], []
            line_fit_mean, line_fit_std = [], []

            # TWs loop
            for _, net in networks.items():
                # Weight values
                line_weight_mean.append(np.mean([e[2]['weight'] for e in net.edges(data=True)]))
                line_weight_std.append(np.std([e[2]['weight'] for e in net.edges(data=True)]))
                # Degree values
                line_degree_mean.append(np.mean([net.in_degree(n) for n in net.nodes]))
                line_degree_std.append(np.std([net.in_degree(n) for n in net.nodes]))
                # Node values
                line_fit_mean.append(np.mean([e[1]['fit'] for e in net.nodes(data=True)]))
                line_fit_std.append(np.std([e[1]['fit'] for e in net.nodes(data=True)]))

            node_degree_mean.append(line_degree_mean)
            node_degree_std.append(line_degree_std)
            execs_weight_mean.append(line_weight_mean)
            execs_weight_std.append(line_weight_std)
            node_fit_mean.append(line_fit_mean)
            node_fit_std.append(line_fit_std)

        node_degree_mean = np.mean(node_degree_mean, axis=0)
        node_degree_std = np.mean(node_degree_std, axis=0)
        execs_weight_mean = np.mean(execs_weight_mean, axis=0)
        execs_weight_std = np.mean(execs_weight_std, axis=0)
        node_fit_mean = np.mean(node_fit_mean, axis=0)
        node_fit_std = np.mean(node_fit_std, axis=0)

        res_nets[alg_name] = [node_degree_mean, execs_weight_mean, node_fit_mean, node_degree_std,
                              execs_weight_std, node_fit_std]
    return res_nets


def plot_swarm_curves(res_swarm, show_std=False, log=False):
    plt.rcParams.update({'font.size': 40})
    num_alg = len(res_swarm)
    fig, axes = plt.subplots(nrows=(num_alg // 4), ncols=4, figsize=(12 * 4, 10 * (num_alg // 4)))
    idx = 0

    for alg_name, values in res_swarm.items():
        means = values[:len(values) // 2]
        stds = values[len(values) // 2:]
        for n, color, y in eczip(means, cmap='tab10', start=0):
            if show_std:
                std_u = y - stds[n]
                std_l = y + stds[n]
                norm_std = [max([np.abs(std_u[i]), np.abs(std_l[i])]) for i in range(len(y))]
                axes[idx // 4][idx % 4].plot(y / max(norm_std), color=color, linewidth=5.0)
                axes[idx // 4][idx % 4].fill_between(range(len(y)), std_u / max(norm_std), std_l / max(norm_std),
                                                     color=color, alpha=0.3)
            else:
                axes[idx // 4][idx % 4].plot(y / max(y), color=color, linewidth=8.0)

            if log:
                axes[idx // 4][idx % 4].set_yscale('log')

            axes[idx // 4][idx % 4].set_title(alg_name)
        idx += 1

    plt.legend(['Best Cost', 'AVG Cost', 'Current Best Cost', 'Current Worst Cost', 'AVG Dist to Best'],
               loc='center left', bbox_to_anchor=(1, 0.5))


def plot_network_curves(res_nets, show_std=False, log=True):
    plt.rcParams.update({'font.size': 40})
    num_alg = len(res_nets)

    fig, axes = plt.subplots(nrows=(num_alg // 4), ncols=4, figsize=(12 * 4, 10 * (num_alg // 4)))
    idx = 0

    titles = ["Node Degree", "Edge Weight", "Node Value"]
    for alg_name, values in res_nets.items():
        means = values[:len(values) // 2]
        stds = values[len(values) // 2:]
        for n, color, y in eczip(means, cmap='tab10', start=0):
            if show_std:
                std_u = y - stds[n]
                std_l = y + stds[n]
                norm_std = [max([np.abs(std_u[i]), np.abs(std_l[i])]) for i in range(len(y))]
                axes[idx // 4][idx % 4].plot(y, color=color, linewidth=5.0)
                axes[idx // 4][idx % 4].fill_between(range(len(y)), std_u, std_l, color=color, alpha=0.3)
            else:
                axes[idx // 4][idx % 4].plot(y, linewidth=8.0)
            if log:
                axes[idx // 4][idx % 4].set_yscale('log')
            axes[idx // 4][idx % 4].set_title(alg_name)

        idx += 1
    plt.legend(titles, fontsize=20, loc='center left', bbox_to_anchor=(1, 0.5))


def diverging_colors(N, cmap=None):
    """
    Make a list of N colors from a color map.
    """
    import matplotlib.cm as mcm
    sm = mcm.ScalarMappable(cmap=cmap)
    return sm.to_rgba(range(N))


def eczip(*args, start=0, step=1, **kw):
    """A combination of enumerate and zip-with-diverging-colors:

    eczip(list1, list2) = [(0, color0, list1[0], list2[0]), (1, color1, list1[1], list2[1]), ...]
    
    Useful for iterating over lists to plot:
    
    >>> for n, color, data in eczip(datasets):
    ...     plt.plot(data, color=color, label='Plot %s' % n)
    """

    args = [list(a) for a in args]
    N = min([len(a) for a in args])

    return zip(range(start, (N + start) * step, step),
               diverging_colors(N, **kw), *args)
