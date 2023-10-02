import sys
from collections import Counter
from math import log

import networkx as nx
import numpy as np
from scipy.stats import entropy
import matplotlib.pyplot as plt


def get_unique_path_lengths(graph, paths=None):
    if paths is None:
        paths = list(nx.all_pairs_dijkstra_path_length(graph))

    unique_path_lengths = set()
    for starting_node, dist_dict in paths:
        unique_path_lengths |= set(dist_dict.values())
    unique_path_lengths = sorted(list(unique_path_lengths))
    return unique_path_lengths


def get_portrait(graph):
    diameter = nx.diameter(graph.to_undirected())
    N = graph.number_of_nodes()
    # B indices are 0...dia x 0...N-1:
    B = np.zeros((diameter + 1, N))

    max_path = 1
    adj = graph.adj
    for starting_node in graph.nodes():
        nodes_visited = {starting_node: 0}
        search_queue = [starting_node]
        d = 1
        while search_queue:
            next_depth = []
            extend = next_depth.extend
            for n in search_queue:
                l = [i for i in adj[n] if i not in nodes_visited]
                extend(l)
                for j in l:
                    nodes_visited[j] = d
            search_queue = next_depth
            d += 1

        node_distances = nodes_visited.values()
        max_node_distances = max(node_distances)

        curr_max_path = max_node_distances
        if curr_max_path > max_path:
            max_path = curr_max_path

        # build individual distribution:
        dict_distribution = dict.fromkeys(node_distances, 0)
        for d in node_distances:
            dict_distribution[d] += 1
        # add individual distribution to matrix:
        for shell, count in dict_distribution.items():
            B[shell][count] += 1

        # HACK: count starting nodes that have zero nodes in farther shells
        max_shell = diameter
        while max_shell > max_node_distances:
            B[max_shell][0] += 1
            max_shell -= 1

    return B[:max_path + 1, :]


def pad_portraits_to_same_size(b1, b2):
    ns, ms = b1.shape
    nl, ml = b2.shape

    # Bmats have N columns, find last *occupied* column and trim both down:
    last_col_1 = max(np.nonzero(b1)[1])
    last_col_2 = max(np.nonzero(b2)[1])
    last_col = max(last_col_1, last_col_2)
    b1 = b1[:, :last_col + 1]
    b2 = b2[:, :last_col + 1]

    big_b1 = np.zeros((max(ns, nl), last_col + 1))
    big_b2 = np.zeros((max(ns, nl), last_col + 1))

    big_b1[:b1.shape[0], :b1.shape[1]] = b1
    big_b2[:b2.shape[0], :b2.shape[1]] = b2

    return big_b1, big_b2


def zeros(shape, tCode=None):
    try:
        return np.zeros(shape, dtype=tCode)
    except TypeError:
        return np.zeros(shape, typecode='fd')  # hardwired to float


def graph_or_portrait(x):
    if isinstance(x, (nx.Graph, nx.DiGraph)):
        return get_portrait(x)
    return x


def element_wise_log(mat):
    new_mat = zeros(mat.shape, tCode=float)
    i = 0
    for row in mat:
        j = 0
        for e in row:
            if e != 0:
                new_mat[i, j] = log(e + 1)
            else:
                new_mat[i, j] = 0
            j += 1
        i += 1
    return new_mat


def create_matrix_plot(o_mat, title, **kwargs):
    kwargs['interpolation'] = 'nearest'
    origin = kwargs.get('origin', 1)
    kwargs['origin'] = 'lower'
    showColorBar = kwargs.get('showColorBar', False)
    if "showColorBar" in kwargs: kwargs.pop("showColorBar")
    logColors = kwargs.get('logColors', False)
    if "logColors" in kwargs: kwargs.pop("logColors")
    ifShow = kwargs.get('show', False)
    if "show" in kwargs: kwargs.pop("show")
    fileName = kwargs.get('fileName', None)
    if "fileName" in kwargs: kwargs.pop("fileName")

    mat = o_mat.copy()  # don't modify original matrix
    if logColors: mat = element_wise_log(mat)

    cmap = plt.cm.seismic
    cmap.set_under(color='white')

    if "vmax" not in kwargs:
        kwargs['vmax'] = float(mat[origin:, origin:].max())
        kwargs['vmin'] = float(mat[origin:, origin:].min() + sys.float_info.epsilon)
        kwargs['cmap'] = cmap

    ax = plt.axes()  # [.05,.05,.9,.9])
    ax.xaxis.tick_top()
    h = plt.imshow(mat, **kwargs)
    plt.title(title)
    plt.axis('tight')
    # ax.set_xlim((origin, 10))
    ax.set_ylim((mat.shape[0], origin))

    if showColorBar: plt.colorbar()

    if fileName is not None:
        plt.savefig(fileName)

    plt.close('all')

    return h


def portrait_divergence(g, h):
    bg = graph_or_portrait(g)
    bh = graph_or_portrait(h)
    bg, bh = pad_portraits_to_same_size(bg, bh)

    l, k = bg.shape
    v = np.tile(np.arange(k), (l, 1))

    xg = bg * v / (bg * v).sum()
    xh = bh * v / (bh * v).sum()

    # flatten distribution matrices as arrays:
    p = xg.ravel()
    q = xh.ravel()

    # lastly, get JSD:
    m = 0.5 * (p + q)
    kld_pm = entropy(p, m, base=2)
    kld_qm = entropy(q, m, base=2)
    jsd_pq = 0.5 * (kld_pm + kld_qm)

    return jsd_pq, bg, bh


def weighted_portrait(g, paths=None, bin_edges=None):
    if paths is None:
        paths = list(nx.all_pairs_dijkstra_path_length(g))

    if bin_edges is None:
        unique_path_lengths = get_unique_path_lengths(g, paths=paths)
        sampled_path_lengths = np.percentile(unique_path_lengths, np.arange(0, 101, 1))
    else:
        sampled_path_lengths = bin_edges
    upl = np.array(sampled_path_lengths)

    l_s_v = []
    for i, (s, dist_dict) in enumerate(paths):
        distances = np.array(list(dist_dict.values()))
        s_v, e = np.histogram(distances, bins=upl)
        l_s_v.append(s_v)
    m = np.array(l_s_v)

    b = np.zeros((len(upl) - 1, g.number_of_nodes() + 1))
    for i in range(len(upl) - 1):
        col = m[:, i]  # ith col = numbers of nodes at d_i <= distance < d_i+1
        for n, c in Counter(col).items():
            b[i, n] += c

    return b


def portrait_divergence_weighted(g, h, bins=None, bin_edges=None):
    # get joint binning:
    paths_g = list(nx.all_pairs_dijkstra_path_length(g))
    paths_h = list(nx.all_pairs_dijkstra_path_length(h))

    # get bin_edges in common for G and H:
    if bin_edges is None:
        if bins is None:
            bins = 1
        upl_g = set(get_unique_path_lengths(g, paths=paths_g))
        upl_h = set(get_unique_path_lengths(h, paths=paths_h))
        unique_path_lengths = sorted(list(upl_g | upl_h))
        bin_edges = np.percentile(unique_path_lengths, np.arange(0, 101, bins))

    # get weighted portraits:
    bg = weighted_portrait(g, paths=paths_g, bin_edges=bin_edges)
    bh = weighted_portrait(h, paths=paths_h, bin_edges=bin_edges)

    return portrait_divergence(bg, bh)
