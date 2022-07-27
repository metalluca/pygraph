from collections import deque

class Vertex:
    def __init__(self, id: int, balance=None, *args, **kwargs):
        self.id = id
        self.balance = balance
"""
class Edge:
    def __init__(self, u: Vertex, v: Vertex, weight=None, capacity=None, *args, **kwargs):
        self.u = u
        self.v = v
        self.weight = weight
        self.capacity = capacity
"""        
class Graph:
    
    def __init__(self, is_directed=False, *args, **kwargs):
        self.V = 0
        self.E = 0
        self.graph = dict()
        self.is_directed = is_directed
    
    def print(self):
        # Info: indices 
        vertex, weight = 0, 1
        print({v.id: [(edge[vertex].id, edge[weight]) for edge in edges] for v, edges in self.graph.items()})
        return None
    
    def add_vertex(self, v_id: int):
        if v_id in [v.id for v in self.graph.keys()]:
            #print("Vertex already in graph.")
            return 1
        else:
            self.V += 1
            self.graph[Vertex(v_id)] = []
            return 0
    
    def get_vertex(self, v_id: int):
        for v in self.graph.keys():
            if v.id == v_id:
                return v
        raise KeyError(f"Vertex with id {v_id} not found in graph.")

    def get_edge(self):
        pass
            
    def add_edge(self, src_id: int, dest_id:int, weight=1.0):
        
        # ? Adding edge with vertices not in the graph?
        # ? Instead of adding edges with vertex objects, add them by id for more usability?
        
        # TODO: Case: Vertices not in graph
        
        v1 = self.get_vertex(src_id)
        v2 = self.get_vertex(dest_id)
        
        self.graph[v1].append((v2, weight))
        if not self.is_directed:
            self.graph[v2].append((v1, weight))
        self.E += 1
        # TODO: How to implement Edge into basic graph structure?
        # TODO: This approach is vertex based 
        # TODO: I guess one may choose between the approaches.
        # TODO: For later problems like the flow-networks another graph strucute might be necessary (edge list)
    
    def build_from_text(self, path):
        """ 
        """
        with open(path) as f:
            lines = f.readlines()
            for idx, line in enumerate(lines):
                if idx == 0:
                    continue 
                
                src_id, dest_id = list(map(int, line.strip("\t\n").split()))
                
                self.add_vertex(src_id)
                self.add_vertex(dest_id)
                
                src_vertex = self.get_vertex(src_id)
                dest_vertex = self.get_vertex(dest_id)
                
                self.add_edge(src_vertex.id, dest_vertex.id)
                
        return None
                
    def breadth_first_search(self, v_start_id, visited=set(), *args):
        """
        IN: starting vertex
        OUT: visited set
        
        TODO: Add tests cases!
        """
        g_bfs = Graph()
        v_start = self.get_vertex(v_start_id)
        
        queue = deque([v_start])
        visited.add(v_start)
        
        while queue:
            curr_v = queue.popleft()
            g_bfs.add_vertex(curr_v.id)
            for neighbour, _ in self.graph[curr_v]:
                if neighbour not in visited:
                    queue.append(neighbour)
                    visited.add(neighbour)
                    g_bfs.add_vertex(neighbour.id)
                    g_bfs.add_edge(curr_v.id, neighbour.id)
        return g_bfs
    
    def depth_first_search(self, v_start_id, visited=set(), *args):
        """
        IN: starting vertex, set of traversed edges
        OUT: None
        
        TODO: Add tests cases!
        """
        v_start = self.get_vertex(v_start_id)
        visited.add(v_start)

        for neighbour, _ in self.graph[v_start]:
            if neighbour not in visited:
                self.depth_first_search(neighbour.id, visited)
                
    def get_connected_component(self, v: Vertex, visited: set):
        result = []
        queue = deque([v])
        while queue:
            vertex = queue.popleft()
            visited.add(vertex.id)
            result.append(vertex)
            for neighbour, _ in self.graph[vertex]:
                if neighbour.id not in visited:
                    queue.append(neighbour)         
        return result, visited  
    
    def get_all_connected_components(self):
        visited = set()
        result = []
        for vertex in self.graph.keys():
            if vertex.id not in visited:
                component, visited = self.get_connected_component(vertex, visited)
                result.append(component)
        return result   
    
    def prim(self):
        pass
    def kruskal(self):
        pass

def get_connected_components(G: Graph):
    def get_connected_component(v: Vertex, visited: set):
        result = []
        queue = deque([v])
        while queue:
            vertex = queue.popleft()
            visited.add(vertex.id)
            result.append(vertex)
            for neighbour, _ in G.graph[vertex]:
                if neighbour.id not in visited:
                    queue.append(neighbour)         
        return result, visited

    visited = set()
    result = []
    for vertex in G.graph.keys():
        if vertex.id not in visited:
            component, visited = G.get_connected_component(vertex, visited)
            result.append(component)
    return result 