import unittest
from graph import Graph, get_connected_components

class TestComponents(unittest.TestCase):

    def test_graph1(self):
        g = Graph()
        g.build_from_text("sample_graphs/Graph1.txt")
        no_components = len(get_connected_components(g))
        self.assertEqual(no_components, 2)
        
    def test_graph2(self):
        g = Graph()
        g.build_from_text("sample_graphs/Graph2.txt")
        no_components = len(get_connected_components(g))
        self.assertEqual(no_components, 4)
        
    def test_graph3(self):
        g = Graph()
        g.build_from_text("sample_graphs/Graph3.txt")
        no_components = len(get_connected_components(g))
        self.assertEqual(no_components, 4)
        
if __name__ == '__main__':
    unittest.main()