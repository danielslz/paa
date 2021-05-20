# pip install networkx matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import sys

from heapq import heapify, heappop
from random import randint


class Heap():
    # data format: [min_weight, node_index, parent]
    heap = []

    def init(self, initial):
        self.heap = initial
        self.rebuild()

    def rebuild(self):
        heapify(self.heap)

    def pop(self):
        return heappop(self.heap)
    
    def size(self):
        return len(self.heap)
    
    def contains(self, node_index):
        for e in self.heap:
            if e[1] == node_index:
                return True
        return False
    
    def update(self, node_index, new_value):
        for i, e in enumerate(self.heap):
            if e[1] == node_index:
                self.heap[i] = new_value
                break
        self.rebuild()
    
    def get(self, node_index):
        for e in self.heap:
            if e[1] == node_index:
                return e
        return None


def build_graph(num_nodes, disconnected=False):
    num_edges = randint(num_nodes, num_nodes * 3)
    top_weight = randint(num_nodes//4, num_nodes//2)

    G = nx.Graph()
    # add nodes
    for i in range(num_nodes):
        G.add_node(i+1)
    # add edges
    if disconnected:
        break_node = randint(1, num_nodes)
        edges_tree_1 = num_edges // break_node
        edges_tree_2 = num_edges - edges_tree_1
        temp_1 = min(edges_tree_1, edges_tree_2)
        temp_2 = max(edges_tree_1, edges_tree_2)
        if break_node < num_nodes // 2:
            edges_tree_1 = temp_1
            edges_tree_2 = temp_2
        else:
            edges_tree_1 = temp_2
            edges_tree_2 = temp_1
        # build subgraph 1
        for i in range(edges_tree_1):
            u = randint(1, break_node)
            v = randint(1, break_node)
            weight = randint(1, top_weight)
            G.add_edge(u, v, weight=weight, color='k')
        # build subgraph 2
        for i in range(edges_tree_2):
            u = randint(break_node + 1, num_nodes)
            v = randint(break_node + 1, num_nodes)
            weight = randint(1, top_weight)
            G.add_edge(u, v, weight=weight, color='k')
    else:
        for i in range(num_edges):
            u = randint(1, num_nodes)
            v = randint(1, num_nodes)
            while u == v:
                v = randint(1, num_nodes)
            weight = randint(1, top_weight)
            G.add_edge(u, v, weight=weight, color='k')
    
    return G


def plot_graph(graph):
    # pos = nx.spring_layout(graph, k=0.15, iterations=20)
    pos = nx.circular_layout(graph)
    edges = graph.edges()
    colors = [graph[u][v]['color'] for u,v in edges]
    nx.draw(graph, pos, edge_color=colors, with_labels=True)
    labels = nx.get_edge_attributes(graph, 'weight')
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=labels)
    plt.show()


def mst_prim(graph, root_node=1):
    heap = Heap()
    mst = nx.Graph()

    # data format: [min_weight, node_index, parent]
    data = []
    for i in range(graph.number_of_nodes()):
        if i+1 == root_node:
            data.append([0, i+1, -1])
        data.append([sys.maxsize, i+1, -1])

    heap.init(data)

    while heap.size() > 0:
        min_weight, node_index, parent = heap.pop() 
        for u, v, d in graph.edges([node_index], data=True):
            w = d['weight']
            # update neighbors
            if heap.contains(v):
                item = heap.get(v)
                if w < item[0]:  # weight
                    new_value = [w, v, u]
                    heap.update(v, new_value)
        if parent > 0:
            # add node in mst
            mst.add_edge(parent, node_index, weight=min_weight)
            # add blue color to edge
            graph[parent][node_index]['color'] = 'b'
    
    return mst


def mst_weight(mst):
    total = 0
    for u, v, weight in mst.edges.data('weight'):
        total += weight
    return total


# build graph
g = build_graph(10, disconnected=True)

# calculate mst
our_mst = mst_prim(g, root_node=5)
lib_mst = nx.minimum_spanning_tree(g, algorithm='kruskal')

# print result
print(f'Our MST: {list(our_mst)}, weight: {mst_weight(our_mst)}')
print(f'Lib MST: {list(lib_mst)}, weight: {mst_weight(lib_mst)}')

# plot
plot_graph(g)