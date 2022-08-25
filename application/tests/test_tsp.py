import unittest
from application.graph import Graph
from application.tsp import double_tree, tsp_nearest_neighbour

# class Test_shortest_paths(unittest.TestCase):
    
#     decimal_place = 0
    
#     def test_graph_wege1_dijkstra(self):
g = Graph(is_weighted=True)
g.build_from_txt("application/sample_graphs/K_10.txt")
print(double_tree(g))
        