import queue
from re import X
import sys
sys.path.insert(0, 'C:/Users/luca-/OneDrive - Fachhochschule Aachen/Python/projects/pygraph/src')
from pygraph import Graph
from heapq import heappop, heappush


def dijkstra(G: Graph, start: int):
    dist = {i:float("inf") for i in range(G.V)}
    pred = {i:None for i in range(G.V)}
    visited = set()
    queue = [(0, start)]
    dist[start] = 0
    
    while queue:
        _, node = heappop(queue)
        if node in visited:
            continue
        visited.add(node)
        curr_dist = dist[node]
        for neighbour, weight in G.get_adjacent_nodes(node):
            if neighbour in visited:
                continue
            dist_to_neighbour = curr_dist + weight
            if dist_to_neighbour < dist[neighbour]:
                heappush(queue, (dist_to_neighbour, neighbour))
                dist[neighbour] = dist_to_neighbour
                pred[neighbour] = node
    return 0

def bellman_ford(G: Graph, start: int):
    dist = {i:float("inf") for i in range(G.V)}
    pred = {i:None for i in range(G.V)}
    dist[start] = 0
     
    def relax(u, v):
        c = G.adj_mat[u][v]
        if dist[u] + c < dist[v]:
            dist[v] = dist[u] + c
            pred[v] = u
            
    for _ in range(G.V - 1):
        for u in range(G.V):
            for v in range(G.V):
                if G.adj_mat[u][v] != 0:
                    relax(u, v)
                    
    for u in range(G.V):
        for v in range(G.V):
            c = G.adj_mat[u][v]
            if c != 0:
                if dist[u] + c < dist[v]:
                    print("Detected negative cycle")       
    print(dist[0])                    
    return 0       

g = Graph(is_weighted=True, is_directed=True)
g.build_from_txt("../sample_graphs/Wege2.txt")
bellman_ford(g, 2)