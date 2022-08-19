from application.graph import Graph
from heapq import heappop, heappush

def prim(G: Graph):
    """
    G is a weighted Graph, start the starting vertex.
    Computes a minimum spanning tree,
    returns the cost.
    """
    E = G.V-1
    start = 0
    mst_cost, edge_count = 0, 0
    mst_edges = []
    visited = set()
    heap = []
    
    def add_edges(src):
        visited.add(src)
        adj_edges = G.get_adjacent_nodes(src)
        for dest, weight in adj_edges:
            if dest not in visited:
                heappush(heap, (weight, (src, dest)))    
    
    add_edges(start)
    while edge_count < E:
        edge_weight, edge = heappop(heap)
        src, dest = edge
        if dest in visited:
            continue
        mst_edges.append(((src, dest), edge_weight))
        edge_count += 1
        mst_cost += edge_weight
        add_edges(dest)
    return mst_cost


def kruskal(G: Graph):
    """
    Basic implementation of kruskals algorithm using union-find. 
    """
    def find(parent, i):
        if parent[i] == i:
            return i
        return find(parent, parent[i])

    def apply_union(parent, x, y):
        xroot = find(parent, x)
        yroot = find(parent, y)
        parent[xroot] = yroot
        
    E = G.V-1
    mst_edges = []
    mst_cost, edge_count = 0, 0
    i = 0
    parent = [i for i in range(G.V)]
    edges = sorted(G.get_edges_list(), key=lambda item: item[2])
    
    while edge_count < E:
        u, v, w = edges[i]
        i += 1
        x = find(parent, int(u))
        y = find(parent, int(v))
        if x != y:
            edge_count += 1
            mst_cost += w
            mst_edges.append(((u, v), w))
            apply_union(parent, x, y)
    return mst_cost
                    
            
        