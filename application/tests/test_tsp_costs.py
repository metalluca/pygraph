import unittest
from application.graph import Graph
from application.tsp import tsp_nearest_neighbour


g = Graph(is_weighted=True, is_directed=False)
g.build_from_txt("application/sample_graphs/K_10.txt")
tsp_nearest_neighbour(g)