
import time
from collections import defaultdict, deque
from functools import lru_cache
from heapq import heapify, heappop, heappush
from itertools import permutations

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

class Edge:
    def __init__(self, src, dest, weight=1):
        """
        by default the weight is "1" in case of an undirected graph (representation in adjacency matrix)
        """
        self.src = int(src)
        self.dest = int(dest)
        self.weight = np.float16(weight)

class Graph:
    def __init__(self, V=None, is_directed=False, is_weighted=False, is_min_flow_network=False):

        self.V = V
        self.adj_mat = None
        self.costs_mat = None
        self.balances = None
        self.is_directed = is_directed
        self.is_weighted = is_weighted
        self.is_min_flow_network = is_min_flow_network

        if is_min_flow_network and is_directed != is_min_flow_network:
            print("Error flow network must be directed!")
        if V:
            self.adj_mat = np.zeros((self.V, self.V), dtype=np.float16)

    def build_from_txt(self, path: str):
        """
        Build a graph with an edges list from a txt file.
        """
        with open(path) as f:
            self.V = int(f.readline())
            self.adj_mat = np.zeros((self.V, self.V), dtype=np.float16)

            if self.is_min_flow_network:
                self.V += 2
                self.adj_mat = np.zeros((self.V, self.V), dtype=np.float16)
                self.costs_mat = np.zeros((self.V, self.V), dtype=np.float16)
                self.balances = np.zeros((self.V, 1))
                lines = f.readlines()

                for idx, line in enumerate(lines[: self.V - 2]):
                    self.balances[idx] = float(line.strip("\t\n"))
                for line in lines[self.V - 2 :]:
                    src, dest, cost, cap = line.strip("\t\n").split()
                    e = Edge(src, dest, cap)
                    self.add_edge(e)
                    self.costs_mat[int(src)][int(dest)] = float(cost)
                    self.costs_mat[int(dest)][int(src)] = float(cost) * -1

            else:
                for line in f.readlines():
                    if self.is_weighted:
                        src, dest, weight = line.strip("\t\n").split()
                        e = Edge(src, dest, weight)
                        self.add_edge(e)
                    else:
                        src, dest = line.strip("\t\n").split()
                        e = Edge(src, dest, 1)
                        self.add_edge(e)
                        
        return Graph

    def show_graph(self, show_weights=True) -> None:
        """
        Visualize the graph with networkx.
        """
        g = nx.DiGraph()
        edges = self.get_edges_list()
        u, v, c = 0, 1, 2
        for edge in edges:
            g.add_edge(edge[u], edge[v], weight=edge[c])
        pos = nx.spring_layout(g)
        graph_show = nx.draw(
            g,
            with_labels=True,
            pos=pos,
            node_size=800,
            node_color="blue",
            font_color="white",
        )
        labels = nx.get_edge_attributes(g, "weight")
        nx.draw_networkx_edge_labels(g, pos=pos, edge_labels=labels)
        plt.show()
        return None

    def get_cycles(self) -> None:
        """
        Prints all cycles and the summed weight of the edges.
        """
        g = nx.DiGraph()
        edges = self.get_edges_list()
        u, v, c = 0, 1, 2
        for edge in edges:
            g.add_edge(edge[u], edge[v], weight=edge[c])
        cs = nx.simple_cycles(g)
        for c in cs:
            print(f"{c} cost: {self.get_cost_of_cycle(tuple(c))}")

    def add_edge(self, e: Edge) -> None:  # u, v, c=1) -> None:
        """
        Method to store the graph either as adj. matrix and adj. list
        """
        self.adj_mat[e.src][e.dest] = e.weight
        if not self.is_directed:
            self.adj_mat[e.dest][e.src] = e.weight

    def get_ausgrad(self, v) -> int:
        return len([i for i in self.adj_mat[v] if i > 0])

    def get_ingrad(self, v) -> int:
        return len([i for i in self.adj_mat[0:, v] if i > 0])

    def get_edge_weight(self, u, v) -> np.float16:
        """ 
        Returns the weight of edge (u, v) 
        """
        return self.adj_mat[u][v]

    def get_edge(self, u, v) -> Edge:
        """
        Returns edge (u, v)
        """
        return Edge(u, v, self.adj_mat[u][v])

    def get_adjacent_edges(self, v) -> list:
        """
        Returns all edges adjacent to v.
        """
        res = []
        for ix, iy in np.ndindex(self.adj_mat.shape):
            if self.adj_mat[ix][iy] > 0 and ix == v:
                if self.is_weighted:
                    res.append((ix, iy, self.adj_mat[ix][iy]))
                else:
                    res.append((ix, iy))
        return res

    def get_adjacent_nodes(self, v) -> list:
        """
        Return alls adjacent nodes with the respective weights if the graph is weighted
        """
        res = []

        for i in range(self.V):
            if self.adj_mat[v][i]:
                if self.is_weighted:
                    res.append((i, self.adj_mat[v][i]))
                else:
                    res.append(self.adj_mat[v][i])
        return res

    def get_edges_list(self,) -> np.array:
        """
        Returns a list of edges (tuples) from the graph object
        by transforming the adjacency matrix.
        """
        res = []
        for ix, iy in np.ndindex(self.adj_mat.shape):
            if self.adj_mat[ix][iy]:
                res.append((ix, iy, self.adj_mat[ix][iy]))
        return np.array(res)

    def nearest_neighbour(self, return_cycle) -> None:
        """
        Method to compute hamiltonian cycle in the graph using nearest neighbour.
        """
        start = 0
        if start >= self.V:
            return f"{start} not in V of G!"
        visited = {
            x: False for x in range(self.V)
        }  # create empty dict to track visited nodes
        cycle = []  # array to append the visited vertices to
        costs = 0
        i = start  # start node
        while True:
            cycle.append(i)
            # cheapest_edge = self.util_find_cheapest_next(i, start, visited)
            try:
                cheapest_edge = min(
                    [edge for edge in self.adj[i] if not visited[edge[0]]],
                    key=lambda x: x[1],
                )
                visited[i] = True
                i = cheapest_edge[0]  # set the node from the cheapest edge as new node
                costs += cheapest_edge[1]  # add the edgecost
            except:  # this code executes if no minimum edge to an unvisited node was found
                cycle.append(start)  # append starting
                for edge in self.adj[
                    i
                ]:  # !!! -> Kante speichern search for the edge costs to start
                    if edge[0] == start:
                        costs += edge[1]
                        break
                break
                # print(f"{i} to {cheapest_edge[0]} with cost {cheapest_edge[1]}")
        if return_cycle:
            print(f"Cycle: {cycle}")
            print(round(costs, 2))
        else:
            print(round(costs, 2))

    def find(self, parent, i) -> int:
        i = int(i)
        if parent[i] == i:
            return i
        return self.find(parent, parent[i])

    def apply_union(self, parent, x, y) -> None:
        # find parents of parents
        xroot = self.find(parent, x)
        yroot = self.find(parent, y)
        # uniion by equal root
        parent[xroot] = yroot

    def MST_KRUSKAL(self) -> list:
        """
        Compute the MST according to Kruskal's Algorithm.
        Input: None
        Output: list of sets with schema (u, v, c) 
        """
        result = []
        i, e = 0, 0
        edges_list = self.get_edges_list()
        edges_list = edges_list[edges_list[:, 2].argsort()]
        parent = []

        for node in range(self.V):
            parent.append(node)

        while e < self.V - 1:
            # take the first edge
            u, v, w = edges_list[i]

            # increase counter
            i = i + 1
            # check if u,v creates a cycle:
            #    x = root of u
            #    y = root of v
            x = self.find(parent, u)
            y = self.find(parent, v)
            # if x != y -> no cycle
            if x != y:
                e += 1
                # add to mst
                result.append((u, v, w))
                # apply union to acknowledge the used edges
                self.apply_union(parent, x, y)

        # cost = sum([i[2] for i in result])
        # print(result)
        # print(f'Kosten des MST nach Kruskal: {cost}')
        return result

    def DFS_util(self, tree_adj, v, visited) -> None:
        visited.append(v)
        for neighbour, _ in tree_adj[v]:
            if neighbour not in visited:
                self.DFS_util(tree_adj, neighbour, visited)

    def DFS(self, tree_adj, start, print_cycle) -> None:
        """
        Perform depth-first-search on a given subtree
        """
        # v = next(iter(mst))
        visited = list()
        self.DFS_util(tree_adj, start, visited)
        # add starting vertex to form a cycle
        visited.append(start)
        if print_cycle:
            print(f"Cycle: {visited}")
        # compute costs
        costs = 0
        for i in range(len(visited) - 1):
            start = visited[i]
            dest = visited[i + 1]
            for v, cost in self.adj[start]:
                if v == dest:
                    costs += cost
        print(round(costs, 2))

    def double_tree(self, print_cycle) -> None:
        """
         Method to compute a heuristic hamiltonian cycle in the graph.
        """
        start = 0  # random.choice(range(self.V))
        mst_edges = self.MST_KRUSKAL()
        mst_adj = defaultdict(list)  # store as adj-matrix

        for edge in mst_edges:
            mst_adj[edge[0]].append((edge[1], edge[2]))
            mst_adj[edge[1]].append((edge[0], edge[2]))

        return self.DFS(mst_adj, start, print_cycle)

    @lru_cache(maxsize=1000)
    def get_cost_of_edge(self, v, u) -> float:
        """
        Return the cost of an edge v,u
        """

        return self.adj_mat[v][u]

    @lru_cache(maxsize=1000)
    def get_cost_of_cycle(self, nodes: list) -> float:
        """
        Return the sum of a path.
        """
        return sum(
            [
                self.get_cost_of_edge(nodes[i], nodes[i + 1])
                for i in range(len(nodes) - 1)
            ]
    
        ) + self.get_cost_of_edge(nodes[-1], nodes[0])
    def get_cost_of_path(self, nodes: tuple) -> float:
        """
        Return the sum of a path.
        """
        return sum(
            [
                self.get_cost_of_edge(nodes[i], nodes[i + 1])
                for i in range(len(nodes) - 1)
            ]
        ) 

    @lru_cache(maxsize=1000)
    def brute_force_tsp(self):
        """
        Compute the minimum solution for the TSP in a given Graph-object. Returns the shortest !!!cycle!!!
        """
        nodes = range(self.V)
        path = min(
            (
                perm
                for perm in permutations(nodes)
                if perm[1] < perm[-1] and perm[0] == 0
            ),
            key=self.get_cost_of_path,
        )
        cop = self.get_cost_of_cycle(path)

        print(f"Cycle: {path} + {path[0]}")
        print(f"Costs: {cop}")

    def path_from_prev(self, start, dest, parent) -> list:
        path = []
        v = dest
        while parent[v] is not None:
            path.insert(0, v)
            v = parent[v]
        path.insert(0, start)
        return path

    def bellman_ford(self, start, dest, res) -> None:

        """ Old bellman ford to find shortest path.
        The detection of negative cycle is wrong since the node which 
        could be improved distancewise may be out of the negative cycle. This logic
        does not consider this case. """

        t_start = time.time()
        dist = {v: float("inf") for v in range(self.V)}
        prev = {v: None for v in range(self.V)}
        dist[start] = 0

        for _ in range(self.V - 1):
            for u, v, c in self.get_edges_list(res):
                if dist[u] + c < dist[v]:
                    dist[v] = dist[u] + c
                    prev[v] = int(u)  # store prev vertex

        for u, v, c in self.get_edges_list(res):
            if dist[u] + c < dist[v]:
                # cycle detected:
                cycle = []
                visited = set()
                visited.add(v)
                x = v
                while prev[x] not in visited:
                    cycle.insert(0, int(x))
                    visited.add(x)
                    x = prev[x]
                cycle.insert(0, int(v))
                # return f"Detected negative cycle {cycle} with cumulated weight {self.get_cost_of_cycle(tuple(cycle))}"
                return False
        path = self.path_from_prev(dest, prev)
        path.insert(0, start)
        # print(f"Kosten von {start} zu {dest}: {round(dist[dest], 2)}")
        # print(f"Pfad             : {path}")
        # print(f"Laufzeit         : {round(time.time()-t_start, 2)} s")
        return path, prev

    def bellman_ford_v2_legacy(self, start=2, dest=0) -> None:

        """Old function do not use anymore"""

        t_start = time.time()
        dist = {v: float("inf") for v in range(self.V)}
        prev = {v: None for v in range(self.V)}
        dist[start] = 0

        for _ in range(self.V - 1):
            for u, v, c in self.get_edges_list():
                if dist[u] + c < dist[v]:
                    dist[v] = dist[u] + c
                    prev[v] = int(u)  # store prev vertex

        C = -1

        for u, v, c in self.get_edges_list():
            if dist[u] + c < dist[v]:
                C = v
                break

        if C != -1:
            for i in range(self.V):
                C = prev[C]

            cycle = []
            v = C

            while True:
                cycle.append(v)
                if v == C and len(cycle) > 1:
                    break
                v = prev[v]
            cycle.reverse()

            for v in cycle:
                print(v, end=" ")
        print(self.get_cost_of_path(tuple(cycle)))
        return None

    def dijkstra(self, start=2, dest=0) -> None:
        """
        Compute all shortest paths from start to every other vertex in the graph.
        """
        t_start = time.time()
        try:
            if not all((i >= 0 for col in self.adj_mat for i in col)):
                raise ValueError
        except ValueError:
            print(
                "ValueError:\nWeight < 0. Dijkstra's algorithm cannot handle negative weights."
            )
            return None

        queue = [(0, start)]  # cost, start-vertex
        dist = {v: float("inf") for v in range(self.V)}
        prev = {v: None for v in range(self.V)}
        visited = set()
        dist[start] = 0

        while queue:
            _, node = heappop(queue)
            if node in visited:  # zaehler auf visited
                continue
            visited.add(node)
            curr_dist = dist[node]
            for neighbour, neighbour_dist in self.get_adjacent_nodes(node):
                if neighbour in visited:
                    continue
                neighbour_dist += curr_dist
                if neighbour_dist < dist[neighbour]:
                    heappush(queue, (neighbour_dist, neighbour))
                    dist[neighbour] = neighbour_dist
                    prev[neighbour] = node
        path = self.path_from_prev(dest, prev)

        print(f"Kosten von {start} zu {dest}: {round(dist[dest], 2)}")
        print(f"Pfad             : {path}")
        print(f"Laufzeit         : {round(time.time()-t_start, 2)} s")
        return None

    def bfs_util(self, res_cap, start, dest, parent) -> bool:
        """
        BFS the given residual capacities until there is no path
        Returns False if no path is found. 
        """
        visited = set()
        queue = deque([start])
        visited.add(start)

        while queue:
            curr_v = queue.popleft()
            for neighbour, cap in enumerate(res_cap[curr_v]):
                if neighbour not in visited and cap > 0:
                    queue.append(neighbour)
                    visited.add(neighbour)
                    parent[neighbour] = curr_v
            if dest in visited:
                return True
        return dest in visited

    def edmond_karp(self, source, sink):
        """
        Determines the max-flow of the graphs capacities with Edmond Karp-Method 
        and Ford-Fulkerson algorithm.
        Returns the max-flow-network
        """
        parent = {v: None for v in range(self.V)}
        max_flow = 0
        res_cap = self.adj_mat.copy()
        flow_graph = np.zeros(self.adj_mat.shape)
        # while there is a path in the network from source to sink
        while self.bfs_util(res_cap, source, sink, parent):
            path_flow = float("Inf")
            s = sink
            while s != source:
                # find the bootleneck in the path
                path_flow = min(path_flow, res_cap[parent[s]][s])
                s = parent[s]

            # Adding the path flows
            max_flow += path_flow
            # updating the residual capacities of edges
            v = sink
            while v != source:
                u = parent[v]
                res_cap[u][v] -= path_flow
                res_cap[v][u] += path_flow
                flow_graph[v][u] -= path_flow
                flow_graph[u][v] += path_flow
                v = parent[v]
        mask = flow_graph < 0
        flow_graph[mask] = 0
        return flow_graph, res_cap

    def bfs_util_sspf(self, res_cap, b, b_strich, start) -> tuple:
        """ 
        BFS the given residual capacities until there is no path
        Returns False if no path is found. 
        """
        visited = set()
        queue = deque([start])
        visited.add(start)

        while queue:
            curr_v = queue.popleft()
            if b[curr_v] - b_strich[curr_v] < 0:
                return curr_v
            for neighbour, cap in enumerate(res_cap[curr_v]):
                if neighbour not in visited and cap > 0:
                    queue.append(neighbour)
                    visited.add(neighbour)
        return None

    def bellman_ford_negative_cycle(self, start, b_flow_res_cap):
        """Computes the shortes path between two nodes.
        If there are negative cycles, any cycle is returned. """

        t_start = time.time()
        dist = {v: float("inf") for v in range(self.V)}
        prev = {v: None for v in range(self.V)}
        dist[start] = 0

        for _ in range(self.V - 1):
            for u, v, c in self.get_edges_list(b_flow_res_cap):
                if dist[u] + c < dist[v]:
                    dist[v] = dist[u] + c
                    prev[v] = int(u)  # store prev vertex
        C = None
        for u, v, c in self.get_edges_list(b_flow_res_cap):
            if dist[u] + c < dist[v]:
                C = v
                break
        if C:
            for i in range(self.V):
                C = prev[C]
            cycle = []
            visited = set()
            x = C
            while x not in visited:
                visited.add(x)
                cycle.insert(0, int(x))
                x = prev[x]
            cycle.append(cycle[0])
            # print(f"detected cycle: {cycle}")
            return cycle, dist
        else:
            # print("No negative cycle detected")
            return None, dist

    def b_flow_ford_fulkerson(self, source, sink):
        # residual graph of max flow
        b_flow = self.edmond_karp(source, sink)
        return 0
        for ix, iy in np.ndindex(b_flow.shape):
            # removing residual edges
            if not self.adj_mat[ix][iy]:
                b_flow[ix][iy] = 0
            # computing flow on forward edges
            else:
                b_flow[ix][iy] = self.adj_mat[ix][iy] - b_flow[ix][iy]
        return b_flow

    def check_b_flow(self, b_flow):
        if sum(self.balances) != 0:
            return False
            # checking balances
        for v in range(self.V):
            in_flow = sum(b_flow[:, v])
            out_flow = sum(b_flow[v, :])
            if not abs(in_flow - out_flow) == abs(self.balances[v]):
                return False
        return True

    def add_super_v_utl(self):

        sources = [(idx, int(b)) for idx, b in enumerate(self.balances) if b > 0]
        sinks = [(idx, int(b)) for idx, b in enumerate(self.balances) if b < 0]
        s_source = self.V - 2
        self.balances[s_source] = sum([t[1] for t in sources])
        s_sink = self.V - 1
        self.balances[self.V - 1] = sum([t[1] for t in sinks])
        for idx, b in sources:
            self.adj_mat[s_source][idx] = self.balances[idx]
            self.costs_mat[s_source][idx] = 0
            self.balances[idx] = 0
        for idx, b in sinks:
            self.adj_mat[idx][s_sink] = abs(self.balances[idx])
            self.costs_mat[idx][s_sink] = 0
            self.balances[idx] = 0

        return s_source, s_sink

    def sspf(self):
        # check ob balancen ausgeglichen sind
        # flow fuer alle kanten mit positiven gewichten 0
        # flow fuer alle kanten mit negativen gewichten = kapazitaet
        b_flow = np.zeros((self.V - 2, self.V - 2))
        mask = self.costs_mat[: self.V - 2, : self.V - 2] < 0
        b_flow[mask] = self.adj_mat[: self.V - 2, : self.V - 2][mask]

        b_strich = np.zeros((self.V - 2, 1))
        b = self.balances[: self.V - 2]

        # b' anhand neuem flow berechnen (out_flow - in_flow)
        for v in range(self.V - 2):
            in_flow = sum(b_flow[:, v])
            out_flow = sum(b_flow[v, :])
            b_strich[v] = out_flow - in_flow

        res = self.adj_mat[: self.V - 2, : self.V - 2] - b_flow
        for x, y in np.ndindex(res.shape):
            if self.adj_mat[x][y] == b_flow[x][y] and self.adj_mat[x][y] > 0:
                res[y][x] = b_flow[x][y]

        while not np.equal(b, b_strich).all():
            # waehle knoten mit b(v) - b'(v) > 0 als start
            # suche fuer eine beliebige pseudo quelle eine pseudosenke
            # wenn keine gefunden naechste pseudoquelle
            s = None
            t = None
            for v in range(self.V - 2):
                if b[v] - b_strich[v] > 0:
                    s = v
                    # pruefe ob beliebige pseudosenke erreichbar (bfs):
                    t = self.bfs_util_sspf(res, b, b_strich, s)
                    if not t:
                        continue
                    break

            if s is None or t is None:
                print("No pair of pseudo vertices found.")
                return None
            # Kuerzesten Weg suchen
            path, parent = self.bellman_ford(s, t, res)
            # wenn kein weg errichbar:a STOPP -> kein b-fluss
            if not path:
                return f"No Path from {s} to {t} found!"
            # min (minimale kapazitaet des weges bestimmen, b(start) - b'(start),  b'(ziel) - b(ziel))
            min_flow = float("inf")
            v = t
            while v != s:
                min_flow = min(min_flow, res[parent[v]][v])
                v = parent[v]

            p_supply = b[s] - b_strich[s]
            p_demand = b_strich[t] - b[t]
            gamma = min(min_flow, p_supply, p_demand)
            # passe fluss anhand des weges an: vorwaertskante: fluss - min, rueckwaertskante: fluss - min
            u = t
            while u != s:
                res[parent[u]][u] -= gamma
                res[u][parent[u]] += gamma
                u = parent[u]
            b_strich[s] += gamma
            b_strich[t] -= gamma

        flow = self.adj_mat[: self.V - 2, : self.V - 2] - res
        flow = np.where(flow > 0, flow, 0)
        flow_costs = flow * self.costs_mat[: self.V - 2, : self.V - 2]
        print(f"min-cost: {flow_costs.sum()}")

    def cycle_cancelling(self) -> None:
        # add super source, super sink
        capacities = self.adj_mat.copy()
        s_source, s_sink = self.add_super_v_utl()
        b_flow, res = self.edmond_karp(s_source, s_sink)
        # check if the flow network is valid

        if not self.check_b_flow(b_flow):
            return "b flow not valid"
        # residual kanten
        # res = capacities - b_flow
        res = np.where(res > 0, res, 0)
        for x, y in np.ndindex(res.shape):
            if b_flow[x][y]:
                res[y][x] = b_flow[x][y]

        while True:

            cycle, dist = self.bellman_ford_negative_cycle(s_source, res)
            if not cycle:
                for v in dist.keys():
                    if dist[v] == float("inf"):
                        cycle, dist = self.bellman_ford_negative_cycle(v, res)
                    if cycle:
                        break
                if not cycle:
                    break
            # break if no cycle found after checking every node as sourc
            min_flow = float("inf")
            for i in range(len(cycle) - 1):
                u, v = cycle[i], cycle[i + 1]
                min_flow = min(min_flow, res[u][v])

            # Berechnung der Residualkapazitaeten ???
            for i in range(len(cycle) - 1):
                u, v = cycle[i], cycle[i + 1]
                res[v][u] += min_flow
                res[u][v] -= min_flow

        flow = capacities - res
        flow = np.where(flow > 0, flow, 0)
        flow_costs = flow * self.costs_mat
        print(f"min-cost: {flow_costs.sum()}")
    def find_min(self, path: list):
        return min(
            [
                self.get_cost_of_edge(path[i], path[i + 1])
                for i in range(len(path) - 1)
            ]) 

def bfs(G: Graph, start: int, end=None, visited=set()):
    """
    Breadth-first search.
    """
    visited.add(start)
    tree = Graph(G.V, is_weighted=True)
    parent = {v: None for v in range(G.V)}
    queue = deque([start])
    
    while queue:
        curr_v = queue.popleft()
        if end and curr_v == end:
            path = G.path_from_prev(start, end, parent)
            return path 
        for neighbour, weight in G.get_adjacent_nodes(curr_v):
            if neighbour not in visited:
                visited.add(neighbour)
                queue.append(neighbour)
                parent[neighbour] = curr_v
                edge = Edge(curr_v, neighbour, weight)
                tree.add_edge(edge)
    return False if end else tree
    

def dfs(G: Graph, start: int):
    """
    Depth-first search.
    """
    pass