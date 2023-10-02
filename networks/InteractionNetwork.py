import glob
import os
import time

import networkx as nx
import numpy as np


class InteractionNetwork(object):
    def __init__(self, population_size, directed=True, output_dir=""):
        self.pos = nx.spring_layout(nx.complete_graph(population_size), iterations=100)
        self.file_path = output_dir + "/INs/"
        self.__create_dir(self.file_path)
        self.population_size = population_size
        self.net_destruction_steps = 100
        self.nodes_counter = 0
        self.ids_dict = {}

        if directed:
            self.graph = nx.DiGraph()
        else:
            self.graph = nx.Graph()
        self.graph.add_nodes_from(range(population_size))

    @staticmethod
    def __create_dir(path):
        if not os.path.isdir(path):
            os.makedirs(path)
        else:
            files = glob.glob(path + '*')
            for f in files:
                os.remove(f)

    @staticmethod
    def __has_link(g, node1, node2):
        return g.get_edge_data(node1, node2)

    def __get_node_id(self, node):
        if type(node) is int or type(node) is np.int32:
            node = str(node)
        elif type(node) is not str:
            node = str(id(node))

        if node in self.ids_dict:
            node_id = self.ids_dict[node]
        else:
            self.ids_dict[node] = self.nodes_counter
            node_id = self.nodes_counter
            self.nodes_counter += 1
        return node_id

    def __add_link(self, g, node1, node2, weight, fits, pos):
        g.add_edge(node1, node2, weight=weight)
        self.graph.nodes[node1]['fit'] = fits[0]
        self.graph.nodes[node2]['fit'] = fits[1]
        self.graph.nodes[node1]['pos'] = ' '.join(str(d) for d in pos[0])
        self.graph.nodes[node2]['pos'] = ' '.join(str(d) for d in pos[1])

    def add_link(self, node1, node2, weight, fits, pos):
        node1 = self.__get_node_id(node1)
        node2 = self.__get_node_id(node2)
        self.__add_link(self.graph, node1, node2, weight, fits, pos)

    def new_iteration(self, i, best):
        fail = True
        while fail:
            try:
                self.graph.add_node("best")
                self.graph.nodes["best"]['fit'] = best.cost
                self.graph.nodes["best"]['pos'] = ' '.join(str(d) for d in best.pos)
                nx.write_gml(self.graph, self.file_path + "inet_graph_iter{}.gml".format(i))
                fail = False
            except Exception as e:
                print('Failed to create the file. Trying again in 10s', e)
                fail = True
                time.sleep(10)

        if self.graph.is_directed():
            self.graph = nx.DiGraph()
        else:
            self.graph = nx.Graph()

    def save_graphs(self):
        pass


