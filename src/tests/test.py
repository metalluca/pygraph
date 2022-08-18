import sys
sys.path.insert(0, 'C:/Users/luca-/OneDrive - Fachhochschule Aachen/Python/projects/pygraph/src')

import unittest
from pygraph import Graph, prim

class TestMST(unittest.TestCase):

    def test_graph1(self):
        g = Graph(is_weighted=True)
        g.build_from_txt("../../sample_graphs/G_1_2.txt")
        res = prim(g, start=0)
        self.assertAlmostEqual(res, 287.98)
                
       
if __name__ == '__main__':
    unittest.main()