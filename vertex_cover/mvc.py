# pip install networkx matplotlib jgrapht
import gc
import jgrapht
import matplotlib.pyplot as plt
import networkx as nx

from heapq import heapify, heappop
from networkx.algorithms.approximation.vertex_cover import min_weighted_vertex_cover
from time import time


class Heap():
    # data format: [node_degree, node_index]
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


def parse_file(data_file):
	adj_list = []
	with open(data_file) as f:
		num_nodes, num_edges, weighted = map(int, f.readline().split())
		for i in range(num_nodes):
			adj_list.append(map(int, f.readline().split()))
	return adj_list


def create_graph_from_file(data_file):
    adj_list = parse_file(data_file)
    G = nx.Graph()
    for i in range(len(adj_list)):
        for j in adj_list[i]:
            G.add_edge(i + 1, j)
    return G


def plot_graph(graph):
    # pos = nx.spring_layout(graph, k=0.15, iterations=20)
    pos = nx.circular_layout(graph)
    edges = graph.edges()
    # colors = [graph[u][v]['color'] for u,v in edges]
    nx.draw(graph, pos, with_labels=True)
    labels = nx.get_edge_attributes(graph, 'weight')
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=labels)
    plt.show()


def build_heap(graph):
    heap = Heap()
    degree_index = {}

    data = []  # data format: [node_degree, node_index]
    for node in graph.nodes:
        node_index = node
        degree = graph.degree[node_index]
        degree_index[node_index] = degree
        # multiply to -1 for desc order
        data.append([-1 * degree, node_index])
    heap.init(data)

    return heap, degree_index


def minimum_vertex_cover(graph):
    mvc = set()

    edges = set(graph.edges)
    heap, degrees = build_heap(graph)

    while len(edges) > 0:
        # remove node with max degree
        node_degree, node_index = heap.pop() 
        adj = set(graph.edges([node_index]))
        for u, v in adj:
            # remove edge from list
            edges.discard((u, v))
            edges.discard((v, u))

            # update neighbors
            if heap.contains(v):
                new_degree = degrees[v] - 1 
                # update index
                degrees[v] = new_degree
                # update heap
                new_value = [-1 * new_degree, v]
                heap.update(v, new_value)

        # add node in mvc
        mvc.add(node_index)
    
    return mvc


def minimum_vertex_cover_2(graph):
    heap = Heap()
    mvc = set()

    # data format: [node_degree, node_index]
    data = []
    for i in range(graph.number_of_nodes()):
        # multiply to -1 for desc order
        data.append([-1*graph.degree[i+1], i+1])

    heap.init(data)
    edges = set(graph.edges)

    while len(edges) > 0:
        # remove node with max degree
        node_degree, node_index = heap.pop() 
        for u, v in graph.edges([node_index]):
            # remove edge from list
            edges.discard((u, v))
            edges.discard((v, u))
        # add node in mvc
        mvc.add(node_index)
    
    return mvc


# tip 1: update heap at X loops
# tip 2: verify if heap implementation is correct
# tip 3: use heap with dict instead of list
def minimum_vertex_cover_3(graph):
    mvc = set()
    visited = {}

    heap, degrees = build_heap(graph)
    edges = set(graph.edges)

    for node in graph.nodes:
        visited[node] = False
    
    while(len(edges) > 0):
        node_degree, node_index = heap.pop()
        if not visited[node_index]:
            visited[node_index] = True
            mvc.add(node_index)
            # remove edges
            for u, v in graph.edges([node_index]):
                # remove edge from list
                edges.discard((u, v))
                edges.discard((v, u))
                # update degree
                degrees[v] -= 1
                if degrees[v] == 0:
                    visited[v] = True

    return mvc


def nx_to_jgraph(graph):
    g = jgrapht.create_graph(directed=False, weighted=True, allowing_self_loops=False, allowing_multiple_edges=False)
    g.add_vertices_from(list(graph.nodes))
    g.add_edges_from(list(graph.edges))
    return g


# build graph
g = create_graph_from_file('data/star2.graph')
jg = nx_to_jgraph(g)

print(f'No of nodes in graph: {g.number_of_nodes()}')
print(f'No of edges in graph: {g.number_of_edges()}')
print('----')

# calculate mvc
gc.collect()
start = time()
mvc = minimum_vertex_cover(g)
end = time()
print(f'Our minimum vertex cover 1: {len(mvc)}, execution time {end-start}s')

gc.collect()
start = time()
mvc = minimum_vertex_cover_2(g)
end = time()
print(f'Our minimum vertex cover 2: {len(mvc)}, execution time {end-start}s')

gc.collect()
start = time()
mvc = minimum_vertex_cover_3(g)
end = time()
print(f'Our minimum vertex cover 3: {len(mvc)}, execution time {end-start}s')

print('----')

gc.collect()
start = time()
mvc = min_weighted_vertex_cover(g)
end = time()
print(f'NetworkX minimum vertex cover: {len(mvc)}, execution time {end-start}s')

gc.collect()
start = time()
mvc = jgrapht.algorithms.vertexcover.greedy(jg)
end = time()
print(f'Jgrapht greedy minimum vertex cover: {int(mvc[0])}, execution time {end-start}s')

gc.collect()
start = time()
mvc = jgrapht.algorithms.vertexcover.edgebased(jg)
end = time()
print(f'Jgrapht edgebased minimum vertex cover: {int(mvc[0])}, execution time {end-start}s')

gc.collect()
start = time()
mvc = jgrapht.algorithms.vertexcover.clarkson(jg)
end = time()
print(f'Jgrapht clarkson minimum vertex cover: {int(mvc[0])}, execution time {end-start}s')

gc.collect()
start = time()
mvc = jgrapht.algorithms.vertexcover.baryehuda_even(jg)
end = time()
print(f'Jgrapht baryehuda_even minimum vertex cover: {int(mvc[0])}, execution time {end-start}s')

# gc.collect()
# start = time()
# mvc = jgrapht.algorithms.vertexcover.exact(jg)
# end = time()
# print(f'Jgrapht exact minimum vertex cover: {int(mvc[0])}, execution time {end-start}s')

# plot
# plot_graph(g)