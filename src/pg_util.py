"""
This file contains utility functions which make use of algorithms but are not a pure implementation.
"""

from pygraph import Graph
from collections import deque

def get_connected_components(G: Graph()) -> list:
    """
    This functions takes a Graph and returns the connected components.
    """
    def get_connected_component(v: int, visited: set):
        result = []
        queue = deque([v])
        
        while queue:
            curr_v = queue.popleft()
            visited.add(curr_v)
            result.append(curr_v)
            for v in range(G.V):
                if G.adj_mat[curr_v][v] and v not in visited:
                    queue.append(v)    
        return result, visited

    visited = set()
    result = []
    for vertex in range(G.V):
        if vertex not in visited:
            component, visited = get_connected_component(vertex, visited)
            result.append(component)
    return result