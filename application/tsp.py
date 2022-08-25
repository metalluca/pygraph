from collections import deque
from application.graph import Graph
from application.minimum_spanning_trees import prim

def tsp_nearest_neighbour(G: Graph) -> float:
    """
    Implementation of greedy nearest neighbour heuristic to solve the TSP.

    """
    start = 0
    visited = set()
    res = []
    tsp_cost = 0
    res.append(start)
    visited.add(start)
    
    curr_v = start
    
    while(len(visited) < G.V):
        next, cost = min([(v, c) for (v, c) in enumerate(G.adj_mat[curr_v]) 
                         if v not in visited and c != 0],
                         key = lambda x: x[1])
        tsp_cost += cost
        curr_v = next
        visited.add(curr_v)
        res.append(curr_v)
        if visited == G.V:
            break
    res.append(start)
    
    return tsp_cost + G.adj_mat[res[-2]][start]

 
def double_tree(G: Graph) -> float: 
    "DOCSTRING"
    g = Graph(is_weighted=True)
    g.build_from_txt("application/sample_graphs/K_10.txt")
    mst_g = prim(g)
    mst_g.V = G.V
    start = 0
    
    visited = set()
    path = []
    def dfs_rec(v, visited):
       visited.add(v)
       path.append(v)
       for neighbour, _ in mst_g.get_adjacent_nodes(v):
            if neighbour not in visited:
                dfs_rec(neighbour, visited)
    def dfs_it(v):
        stack = deque([v])
        visited = set()
        
        while stack:
            curr_v = stack.pop()
            if curr_v in visited:
                continue
            visited.add(curr_v)
            path.append(curr_v)
            for neighbour, _ in mst_g.get_adjacent_nodes(curr_v):
                if neighbour not in visited:
                    stack.append(neighbour)
    dfs_rec(start, visited)
    res = sum([
                G.get_cost_of_edge(path[i], path[i + 1])
                for i in range(len(path) - 1)
            ]) + G.get_cost_of_edge(path[-1], path[0])
    return res
    # TODO: Implement DFS for the rest of the algorithm