from application.graph import Graph

def tsp_nearest_neighbour(G: Graph):
    
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
 