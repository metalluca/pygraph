import sys
sys.path.insert(0, 'C:/Users/luca-/OneDrive - Fachhochschule Aachen/Python/projects/pygraph/src')

import unittest
from pygraph import Graph
from pg_util import get_connected_components

class TestComponents(unittest.TestCase):

    def test_graph1(self):
        g = Graph()
        g.build_from_txt("../../sample_graphs/Graph1.txt")
        no_components = len(get_connected_components(g))
        self.assertEqual(no_components, 2)
        
    def test_graph2(self):
        g = Graph()
        g.build_from_txt("../../sample_graphs/Graph2.txt")
        no_components = len(get_connected_components(g))
        self.assertEqual(no_components, 4)
        
    def test_graph3(self):
        g = Graph()
        g.build_from_txt("../../sample_graphs/Graph3.txt")
        no_components = len(get_connected_components(g))
        self.assertEqual(no_components, 4)
    """
    def test_graph4(self):
        g = Graph()
        g.build_from_txt("../../sample_graphs/Graph_gross.txt")
        no_components = len(get_connected_components(g))
        self.assertEqual(no_components, 222)

    def test_graph5(self):
        g = Graph()
        g.build_from_txt("../sample_graphs/Graph_ganzgross.txt")
        no_components = len(get_connected_components(g))
        self.assertEqual(no_components, 9560)
        
    def test_graph6(self):
        g = Graph()
        g.build_from_txt("../sample_graphs/Graph_ganzganzgross.txt")
        no_components = len(get_connected_components(g))
        self.assertEqual(no_components, 306)
    """   
if __name__ == '__main__':
    unittest.main()