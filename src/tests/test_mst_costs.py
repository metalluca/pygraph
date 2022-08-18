import sys
sys.path.insert(0, 'C:/Users/luca-/OneDrive - Fachhochschule Aachen/Python/projects/pygraph/src')

import unittest
from pygraph import Graph, prim

class TestMST(unittest.TestCase):
    
    decimal_place = 0
    
    def test_graph_g_1_2(self):
        g = Graph(is_weighted=True)
        g.build_from_txt("../../sample_graphs/G_1_2.txt")
        res = prim(g, start=0)
        self.assertEqual(int(res), 287, self.decimal_place)
        
    def test_graph_g_1_20(self):
        g = Graph(is_weighted=True)
        g.build_from_txt("../../sample_graphs/G_1_20.txt")
        res = prim(g, start=0)
        self.assertEqual(int(res), 36, self.decimal_place)
        
    def test_graph_g_1_200(self):
        g = Graph(is_weighted=True)
        g.build_from_txt("../../sample_graphs/G_1_200.txt")
        res = prim(g, start=0)
        self.assertEqual(int(res), 12, self.decimal_place)
        
    def test_graph_g_10_20(self):
        g = Graph(is_weighted=True)
        g.build_from_txt("../../sample_graphs/G_10_20.txt")
        res = prim(g, start=0)
        self.assertEqual(int(res), 2785, self.decimal_place)
        
    def test_graph_g_1_200(self):
        g = Graph(is_weighted=True)
        g.build_from_txt("../../sample_graphs/G_10_200.txt")
        res = prim(g, start=0)
        self.assertEqual(int(res), 372, self.decimal_place)
        
    def test_graph_g_100_200(self):
        g = Graph(is_weighted=True)
        g.build_from_txt("../../sample_graphs/G_100_200.txt")
        res = prim(g, start=0)
        self.assertEqual(int(res), 27550, self.decimal_place)    
       
if __name__ == '__main__':
    unittest.main()