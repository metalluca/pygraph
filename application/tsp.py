# -*- coding: utf-8 -*- 
"""
   Approaches to solve the Travelling Salesman problem.
   -----------------------------------------------------------------------------------------  
   Implementation consists of the nearest neighbour heuristic, the double tree heuristic and a Brute-Force approach.
   The heuristic solutions are not the optimal but provide a solution in polynomial time.
   Note that these implementations are not efficient.

   Todo: 
       * Brute-Force 
       * Branch-and-Bound 
"""

from collections import deque
from itertools import permutations
from application.graph import Graph
from application.minimum_spanning_trees import prim

def tsp_nearest_neighbour(G: Graph) -> float:
    """Implemented of greedy nearest neighbour heuristic to solve the TSP."""
    start = 0
    visited = set()
    path = []
    tsp_cost = 0
    path.append(start)
    visited.add(start)
    
    curr_v = start
    
    while(len(visited) < G.V):
        next, cost = min([(v, c) for (v, c) in enumerate(G.adj_mat[curr_v]) 
                         if v not in visited and c != 0],
                         key = lambda x: x[1])
        tsp_cost += cost
        curr_v = next
        visited.add(curr_v)
        path.append(curr_v)
        if visited == G.V:
            break
    path.append(start)
    
    return tsp_cost + G.adj_mat[path[-2]][start] #  Append returning edge: end - start to form the tour

 
def tsp_double_tree(G: Graph) -> float: 
    """Implementation of the double tree heuristic to solve the TSP."""
    mst_g = prim(G)
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
    
    return G.get_cost_of_cycle(tuple(path))

def brute_force_tsp(G: Graph):
        """
        Compute the optimal solution for the TSP in a given Graph-object. 
        Returns the costs of the minimal tour.
        Note the complexity of O(n!).
        """
        nodes = range(G.V)
        path = min(
            (
                perm
                for perm in permutations(nodes)
                if perm[1] < perm[-1] and perm[0] == 0
            ),
            key=G.get_cost_of_path,
        )
        tsp_cost = G.get_cost_of_cycle(path)
        return tsp_cost