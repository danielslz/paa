# pip install networkx matplotlib jgrapht
import gc
import jgrapht
import matplotlib.pyplot as plt
import networkx as nx

from heapq import heapify, heappop
from networkx.algorithms.approximation.vertex_cover import min_weighted_vertex_cover
from time import perf_counter
from networkx.algorithms.assortativity.pairs import node_degree_xy

from networkx.generators.atlas import graph_atlas_g


class Heap():
    # data format: [node_degree, node_index]
    heap = []
    hash = dict()

    def init(self, initial):
        self.heap = initial
        for value, index in initial:
            self.hash[index] = value
        self.rebuild()

    def rebuild(self):
        heapify(self.heap)

    def pop(self):
        return heappop(self.heap)

    def contains(self, index):
        return index in self.hash
        
    def update(self, index, value):
        self.hash[index] = value
        for i, e in enumerate(self.heap):
            if e[1] == index:
                self.heap[i] = [value, index]
                break
        self.rebuild()

    def get(self, index):
        return self.hash.get(index)
    
    def size(self):
        return len(self.heap)


DIMACS_GRAPH = 1
SNAP_GRAPH = 2
CS6140_GRAPH = 3

def parse_file(data_file, graph_format):
    adj_list = []
    with open(data_file) as f:
        if graph_format == DIMACS_GRAPH:
            lines = f.readlines()
            for line in lines:
                keys = line.split()
                if keys[0] == 'e':
                    adj_list.append((int(keys[1]), int(keys[2])))
        elif graph_format == SNAP_GRAPH:
            lines = f.readlines()
            for line in lines:
                keys = line.split()
                if keys[0] != '#':
                    adj_list.append((int(keys[0]), int(keys[1])))
        elif graph_format == CS6140_GRAPH:
            num_nodes, num_edges, weighted = map(int, f.readline().split())
            for i in range(num_nodes):
                adj_list.append(map(int, f.readline().split()))
        return adj_list


def create_graph_from_file(data_file, graph_format=CS6140_GRAPH):
    adj_list = parse_file(data_file, graph_format)
    G = nx.Graph()
    if graph_format in (DIMACS_GRAPH, SNAP_GRAPH):
        for a, b in adj_list:
            G.add_edge(a, b)
    elif graph_format == CS6140_GRAPH:
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


def minimum_vertex_cover_pure_greedy(graph):
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
                heap.update(v, -1 * new_degree)

        # add node in mvc
        mvc.add(node_index)

    return mvc


def minimum_vertex_cover_approximation(graph):
    mvc = set()

    edges = set(graph.edges)
    nodes = set(graph.nodes)

    while len(edges) > 0:
        # pick any node
        node = nodes.pop()
        for u, v in graph.edges([node]):
            # remove edge from list
            edges.discard((u, v))
            edges.discard((v, u))
        # add node in mvc
        mvc.add(node)

    return mvc


def get_degrees(graph):
    degrees = {}

    for node in graph.nodes:
        node_index = node
        degree = graph.degree[node_index]
        degrees[node_index] = degree
    
    return degrees


def get_heap(nodes, degrees, visited):
    heap = Heap()
    heap_data = []  # data format: [node_degree, node_index]
    for node in nodes:
        if not visited[node]:
            degree = degrees[node]
            # multiply to -1 for desc order
            heap_data.append([-1 * degree, node])
    heap.init(heap_data)

    return heap


def remove_edges_and_update_degrees(edges_to_remove, edges, degrees, visited):
    for u, v in edges_to_remove:
        # remove edge from list
        edges.discard((u, v))
        edges.discard((v, u))
        # update degree
        degrees[v] -= 1
        if degrees[v] == 0:
            visited[v] = True


def minimum_vertex_cover_hybrid_greedy(graph):
    mvc = set()
    visited = {}

    degrees = get_degrees(graph)
    edges = set(graph.edges)
    nodes = set(graph.nodes)

    # mark node with degree 1, otherwise not visited
    for node in nodes:
        # init status
        visited[node] = False
        if degrees[node] == 1:
            # mark node as visited
            visited[node] = True
            # remove edges and update node degrees
            remove_edges_and_update_degrees(graph.edges([node]), edges, degrees, visited)
 
    # build heap with nodes not visited
    heap = get_heap(nodes, degrees, visited)

    # heap update factor
    heap_update_factor = 1
    total_nodes = heap.size()
    # ratio = total_nodes / len(edges)
    ratio = 0.01
    if len(nodes) > 100:
        heap_update_factor = int(total_nodes * ratio)

    # greedy
    count = 0
    while(len(edges) > 0):
        count += 1
        # verify if must update heap
        if count > heap_update_factor:
            count = 0
            heap = get_heap(nodes, degrees, visited)

        _, node_index = heap.pop()
        if not visited[node_index]:
            visited[node_index] = True
            mvc.add(node_index)
            # remove edges
            remove_edges_and_update_degrees(graph.edges([node_index]), edges, degrees, visited)

    return mvc


def nx_to_jgraph(graph):
    g = jgrapht.create_graph(directed=False, weighted=True,
                             allowing_self_loops=False, allowing_multiple_edges=False)
    g.add_vertices_from(list(graph.nodes))
    g.add_edges_from(list(graph.edges))
    return g


def run(methods, graph, is_jgrapht=False):
    for func, msg in methods:
        gc.collect()
        start = perf_counter()
        mvc = func(graph)
        end = perf_counter()
        if is_jgrapht:
            # Jgrapht result
            print(f'{msg} vertex cover: {int(mvc[0])}, execution time {end-start:0.5f}s')
        else:
            print(f'{msg} vertex cover: {len(mvc)}, execution time {end-start:0.5f}s')


# build graph
# g = create_graph_from_file('data/cs6140/dummy4.graph')
# g = create_graph_from_file('data/dimacs/flat1000_76_0.col', graph_format=DIMACS_GRAPH)
g = create_graph_from_file('data/bhoslib/frb59-26-1.mis', graph_format=DIMACS_GRAPH)
# g = create_graph_from_file('data/snap/p2p-Gnutella08.txt', graph_format=SNAP_GRAPH)


jg = nx_to_jgraph(g)

print(f'No of nodes: {g.number_of_nodes()}')
print(f'No of edges: {g.number_of_edges()}')
print('----')

# calculate mvc
our_methods = [
    (minimum_vertex_cover_approximation, 'Approximation'),
    # (minimum_vertex_cover_pure_greedy, 'Pure greedy'),
    (minimum_vertex_cover_hybrid_greedy, 'Hybrid greedy')
]
run(our_methods, g)

print('-----')

lib_methods = [
    (jgrapht.algorithms.vertexcover.greedy, 'Jgrapht greedy'),
    (jgrapht.algorithms.vertexcover.edgebased, 'Jgrapht edgebased'),
    (jgrapht.algorithms.vertexcover.clarkson, 'Jgrapht clarkson'),
    (jgrapht.algorithms.vertexcover.baryehuda_even, 'Jgrapht baryehuda_even'),
    # (jgrapht.algorithms.vertexcover.exact, 'Jgrapht exact'),
]
run(lib_methods, jg, is_jgrapht=True)


# gc.collect()
# start = perf_counter()
# mvc = min_weighted_vertex_cover(g)
# end = perf_counter()
# print(f'NetworkX minimum vertex cover: {len(mvc)}, execution time {end-start:0.5f}s')


# plot
# plot_graph(g)
