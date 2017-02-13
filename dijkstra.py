# dijkstra's algorithm
"""
1. Assign to every node a distance value. Set it to zero for our initial node 
   and to infinity for all other nodes.
 
2. Mark all nodes as unvisited. Set initial node as current.
 
3. For current node, consider all its unvisited neighbors and calculate their 
   tentative distance (from the initial node). For example, if current node 
   (A) has distance of 6, and an edge connecting it with another node (B) 
   is 2, the distance to B through A will be 6+2=8. If this distance is less 
   than the previously recorded distance (infinity in the beginning, zero 
   for the initial node), overwrite the distance.
 
4. When we are done considering all neighbors of the current node, mark it as 
   visited. A visited node will not be checked ever again; its distance 
   recorded now is final and minimal.
 
5. If all nodes have been visited, finish. Otherwise, set the unvisited node 
   with the smallest distance (from the initial node) as the next "current 
   node" and continue from step 3.
 
 - source: wikipedia http://en.wikipedia.org/wiki/Dijkstra%27s_algorithm
"""
 
nodes = set()
edges = {}
distances = {}
_distance = []
	#print "graph instantiated"
    
def _add_node(value):
  nodes.add(value)
    
def add_edge(from_node, to_node, distance):
  _add_edge(from_node, to_node, distance)
  _add_edge(to_node, from_node, distance)
  _distance.append([from_node, to_node, distance])
  _distance.append([to_node, from_node, distance])
 
def _add_edge(from_node, to_node, distance):
  edges.setdefault(from_node, [])
  edges[from_node].append(to_node)
  distances[(from_node, to_node)] = distance
 
 
def dijkstra(initial_node):
    visited = {initial_node: 0}
    current_node = initial_node
    path = {}
    
    nodes_ = set(nodes)
    
    while nodes_:
        min_node = None
        for node in nodes_:
            if node in visited:
                if min_node is None:
                    min_node = node
                elif visited[node] < visited[min_node]:
                    min_node = node
 
        if min_node is None:
            break
 
        nodes_.remove(min_node)
        cur_wt = visited[min_node]
        
        for edge in edges[min_node]:
            wt = cur_wt + distances[(min_node, edge)]
            if edge not in visited or wt < visited[edge]:
                visited[edge] = wt
                path[edge] = min_node
    
    return visited, path
 
def shortest_path(initial_node, goal_node):
    distances, paths = dijkstra(initial_node)
    route = [goal_node]
 
    while goal_node != initial_node:
        route.append(paths[goal_node])
        goal_node = paths[goal_node]
 
    route.reverse()
    return route

def get_distance(_distance, a, b):
    for d in _distance:
	if d[0]==a and d[1]==b:
	    return d[2]
#exception..

def shortest_distance(src, dst):
    route = shortest_path(src, dst)
    d = 0
    for i in range(len(route)-1):
	d = d + get_distance(_distance, route[i], route[i+1])
    return d
	
	
 
if __name__ == '__main__':
    
    nodes = set(range(1, 7))
    add_edge(1, 2, 7)
    add_edge(1, 3, 9)
    add_edge(1, 6, 14)
    add_edge(2, 3, 10)
    add_edge(2, 4, 15)
    add_edge(3, 4, 11)
    add_edge(3, 6, 2)
    add_edge(4, 5, 6)
    add_edge(5, 6, 9)	
    shortest_distance(1, 5)
    #print g._distance
    print shortest_path(1, 5)
    print shortest_distance(1, 5)
    #assert shortest_path(g, 1, 5) == [1, 3, 6, 5]
    #assert shortest_path(g, 5, 1) == [5, 6, 3, 1]
    #assert shortest_path(g, 2, 5) == [2, 3, 6, 5]
    #assert shortest_path(g, 1, 4) == [1, 3, 4]

