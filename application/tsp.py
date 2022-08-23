from application.graph import Graph
from application.minimum_spanning_trees import prim

def tsp_nearest_neighbour(G: Graph):
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

 
def double_tree(G: Graph): 
     g = Graph(is_weighted=True)
     g.build_from_txt("application/sample_graphs/K_10.txt")
     mst_g = prim(g)
     mst_g.show_graph()
     # TODO: Implement BFS or DFS for the rest of the algorithm