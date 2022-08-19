import sys
sys.path.insert(0, 'C:/Users/luca-/OneDrive - Fachhochschule Aachen/Python/projects/pygraph/src')

import unittest
from pygraph import Graph
from shortest_paths import dijkstra, bellman_ford

class Test_shortest_paths(unittest.TestCase):
    
    decimal_place = 0
    
    def test_graph_wege1_dijkstra(self):
        g = Graph(is_weighted=True, is_directed=True)
        g.build_from_txt("../../sample_graphs/Wege1.txt")
        res = dijkstra(g, start=2)[0]
        self.assertEqual(int(res), 6, self.decimal_place)
        
    def test_graph_wege1_bellman_ford(self):
        g = Graph(is_weighted=True, is_directed=True)
        g.build_from_txt("../../sample_graphs/Wege1.txt")
        res = bellman_ford(g, start=2)[0]
        self.assertEqual(int(res), 6, self.decimal_place)

    def test_graph_wege2_dijkstra(self):
        g = Graph(is_weighted=True, is_directed=True)
        g.build_from_txt("../../sample_graphs/Wege2.txt")
        res = dijkstra(g, start=2)
        self.assertEqual(res, False, self.decimal_place)
        
    def test_graph_wege2_bellman_ford(self):
        g = Graph(is_weighted=True, is_directed=True)
        g.build_from_txt("../../sample_graphs/Wege2.txt")
        res = bellman_ford(g, start=2)[0]
        self.assertEqual(int(res), 2, self.decimal_place)

    def test_graph_wege3_dijkstra(self):
        g = Graph(is_weighted=True, is_directed=True)
        g.build_from_txt("../../sample_graphs/Wege3.txt")
        res = dijkstra(g, start=2)
        self.assertEqual(res, False, self.decimal_place)
        
    def test_graph_wege3_bellman_ford(self):
        g = Graph(is_weighted=True, is_directed=True)
        g.build_from_txt("../../sample_graphs/Wege3.txt")
        res = bellman_ford(g, start=2) 
        self.assertEqual(res, False, self.decimal_place)

if __name__ == '__main__':
    unittest.main()