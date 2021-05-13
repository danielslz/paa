import matplotlib.pyplot as plt
import networkx as nx

from random import randint


def build_graph(num_nodes):
    num_edges = randint(num_nodes, num_nodes * 3)
    top_weight = randint(num_nodes//4, num_nodes//2)

    G = nx.Graph()
    for i in range(num_edges):
        u = randint(1, num_nodes)
        v = randint(1, num_nodes)
        weight = randint(1, top_weight)
        G.add_edge(u, v, weight=weight)
    
    return G

def plot_graph(graph):
    pos = nx.spring_layout(graph)
    nx.draw(graph, pos, with_labels=True)
    labels = nx.get_edge_attributes(graph, 'weight')
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=labels)
    plt.show()


g = build_graph(20)

plot_graph(g)