# This file is your main submission that will be graded against. Only copy-paste
# code on the relevant classes included here from the IPython notebook. Do not
# add any classes or functions to this file that are not part of the classes
# that we want.
from __future__ import division
import math
from osm2networkx import *
import random
import pickle
import sys
from collections import deque
# Comment the next line when submitting to bonnie
# import matplotlib.pyplot as plt

# Implement a heapq backed priority queue (accompanying the relevant question)
import heapq
from scipy.spatial import distance
import Queue
class PriorityQueue():
    
    # HINT look up the module heapq.

    def __init__(self):
        self.queue = []
        self.current = 0

    def next(self):
        if self.current >=len(self.queue):
            self.current
            raise StopIteration
    
        out = self.queue[self.current]
        self.current += 1

        return out

    def isEmpty(self):
        return len(self.queue) == 0

    def __len__(self):
        return len(self.queue)

    def pop(self):
        if self.queue:
            return heapq.heappop(self.queue)
        return None

    def peek(self):
        if len(self.queue)==0:
            return (0,None)
        return self.queue[0]

    def get(self,nodeId):
        for index, (c, n) in enumerate(self.queue):
            if nodeId == n[-1]:
                return self.queue[index]
        return []
    def remove(self, nodeId):
        for index, (c, n) in enumerate(self.queue):
            if nodeId == n[-1]:
                temp = self.queue[index]
                del self.queue[index]
                return temp
                break
        heapq.heapify(self.queue)
    
    def __iter__(self):
        return self

    def __str__(self):
        return 'PQ:[%s]'%(', '.join([str(i) for i in self.queue]))

    def append(self, node):
        heapq.heappush(self.queue,node)

    def __contains__(self, key):
        self.current = 0
        for v, n in self.queue:
            if key == n[len(n)-1]:
                return True
        return  False

    def __eq__(self, other):
        self.current = 0
        return self == other

    __next__ = next

def get_path_cost(graph,path):
    path_sum = 0
    if path and len(path)>1:
        for i in range(0,len(path)-1):
            u = path[i]
            v = path[i+1]
            path_sum += graph[u][v]['weight']

    return path_sum

def graph_search(graph, start, goal, alg):

    frontier = PriorityQueue()
    # frontier = initial state
    frontier.append((0,[start]))

    # if frontier is empty return []
    if start == goal or not graph.has_node(start) or not graph.has_node(goal):
        return []

    while not frontier.isEmpty(): # while nodes left in frontier:
        # path = remove_choice(frontier)

        cost, path = frontier.pop() # priority queue chooses shortest path
        # print "popped: " , path
        curr_node = path[-1]
        # if node is goal return path
        if curr_node == goal:
            return path

        # add node to explored set
        graph.explored_nodes.add(curr_node)

        # for each neighbor, if it hasnt been explored
        #  append it to the current path
        #  add the new path to the frontier
        # print "explored: " , graph.get_explored_nodes()

        for neighbor in graph[curr_node]: #this check breaks CFS, because the cheapest neighbor has already been explored.
            new_path = None
            explored = graph.get_explored_nodes()
            new_path = copy.deepcopy(path)
            new_path.append(neighbor)
            total_cost = cost + graph[curr_node][neighbor]['weight']
            if neighbor not in explored and neighbor not in frontier:
                frontier.append((total_cost,new_path))
            elif neighbor in frontier:
                for c, p in frontier:
                    node = p[-1]
                    if neighbor == node:
                        if total_cost < c:
                            frontier.remove(neighbor)
                            frontier.append((total_cost,new_path))
                        break

    return []

#Warmup exercise: Implement breadth-first-search
def breadth_first_search(graph, start, goal):
    frontier = PriorityQueue()
    # frontier = initial state
    frontier.append((0,[start]))

    if start == goal or not graph.has_node(start) or not graph.has_node(goal):
        return []

    while frontier: # while nodes left in frontier:
        # path = remove_choice(frontier)
        if frontier.isEmpty():
            return []
        cost, path = frontier.pop() # breadth first search chooses shortest path
        # check the end of the current path
        curr_node = path[-1]

        if curr_node == goal:
            return path

        if curr_node not in graph.get_explored_nodes():
            # add node to explored set
            graph.explored_nodes.add(curr_node)

            # for each neighbor, if it hasnt been explored
            #  append it to the current path
            #  add the new path to the frontier
            for neighbor in graph[curr_node]: #this check breaks CFS, because the cheapest neighbor has already been explored.
                new_path = None
                # if node is goal return path
                if neighbor == goal:
                    new_path = path.append(neighbor)
                    return path

                if neighbor not in graph.get_explored_nodes():
                    new_path = copy.deepcopy(path)
                    new_path.append(neighbor)
                    total_cost = len(new_path)
                    frontier.append((total_cost,new_path))


#Warmup exercise: Implement uniform_cost_search
def uniform_cost_search(graph, start, goal):
    return graph_search(graph, start, goal, 1)

# Warmup exercise: Implement A*
def null_heuristic(graph, v, goal ):
    return 0

# Warmup exercise: Implement the euclidean distance heuristic
def euclidean_dist_heuristic(graph, v, goal):
    if v == goal:
        return 0

    if 'pos' in graph.node[v]:
        posv = graph.node[v]['pos']
        posgoal = graph.node[goal]['pos']
    else:
        posv = graph.node[v]['position']
        posgoal = graph.node[goal]['position']

    return distance.euclidean(posv,posgoal)

# Warmup exercise: Implement A* algorithm
def a_star(graph, start, goal, heuristic=euclidean_dist_heuristic):
    frontier = PriorityQueue()
    # frontier = initial state
    frontier.append((0,[start]))

    # if frontier is empty return []
    if start == goal or not graph.has_node(start) or not graph.has_node(goal):
        return []

    while not frontier.isEmpty(): # while nodes left in frontier:
        # path = remove_choice(frontier)

        cost, path = frontier.pop() # priority queue chooses shortest path
        # print "popped: " , path
        curr_node = path[-1]
        # if node is goal return path
        if curr_node == goal:
            return path

        # add node to explored set
        graph.explored_nodes.add(curr_node)

        # for each neighbor, if it hasnt been explored
        #  append it to the current path
        #  add the new path to the frontier
        # print "explored: " , graph.get_explored_nodes()

        for neighbor in graph[curr_node]: #this check breaks CFS, because the cheapest neighbor has already been explored.
            new_path = None
            explored = graph.get_explored_nodes()
            new_path = copy.deepcopy(path)
            new_path.append(neighbor)
            total_cost = get_path_cost(graph,new_path)
            total_cost += heuristic(graph,neighbor,goal)
            if neighbor not in explored and neighbor not in frontier:
                frontier.append((total_cost,new_path))
            elif neighbor in frontier:
                for c, p in frontier:
                    node = p[-1]
                    if neighbor == node:
                        if total_cost < c:
                            frontier.remove(neighbor)
                            frontier.append((total_cost,new_path))
                        break

    return []

# Exercise 1: Bidirectional Search
def bidirectional_ucs(graph, start, goal):
    frontier_end = PriorityQueue()
    frontier_start = PriorityQueue()
    frontier_start.append((0,[start]))
    frontier_end.append((0,[goal]))
    start_explored = []
    end_explored = []
    best_path = []
    best_cost = float("inf")

     # if frontier is empty return []
    if start == goal or not graph.has_node(start) or not graph.has_node(goal):
        return []

    while frontier_start and frontier_end:

        if frontier_start.peek()[0]  + frontier_end.peek()[0] >= best_cost:
            return best_path

        elif frontier_start.peek()[0] <= frontier_end.peek()[0]:
            if not frontier_start.isEmpty():
                start_cost, start_path = frontier_start.pop()
                start_node = start_path[-1]
            if start_node == goal:
                if best_cost > start_cost:
                    best_cost = start_cost
                    best_path = start_path
            elif start_node in frontier_end:
                end_cost, end_path = frontier_end.get(start_node)
                reversedend = copy.deepcopy(end_path)
                reversedend.reverse()
                if best_cost > end_cost + start_cost:
                    best_cost = end_cost + start_cost
                    best_path = start_path[0:-1] + reversedend

             # add node to explored set
            start_explored.append(start_node)
            graph.explored_nodes.add(start_node)
            for neighbor in graph[start_node]:
                new_path = None
                new_path = copy.deepcopy(start_path)
                new_path.append(neighbor)
                total_cost = start_cost + graph[start_node][neighbor]['weight']
                if neighbor not in start_explored and neighbor not in frontier_start:
                    frontier_start.append((total_cost,new_path))
                elif neighbor in frontier_start:
                    for c, p in frontier_start:
                        node = p[-1]
                        if neighbor == node:
                            if total_cost < c:
                                frontier_start.remove(neighbor)
                                frontier_start.append((total_cost,new_path))
                            break
        else:
            if not frontier_end.isEmpty():
                end_cost, end_path = frontier_end.pop()
                end_node = end_path[-1]
                reversedend = copy.deepcopy(end_path)
                reversedend.reverse()
            if end_node == start:
                if best_cost > end_cost:
                    best_cost = end_cost
                    best_path = reversedend
            elif end_node in frontier_start:
                c, start_path = frontier_start.remove(end_node)
                first = start_path[0:-1]
                reversedend = copy.deepcopy(end_path)
                reversedend.reverse()
                if best_cost > end_cost+c:
                    best_cost = end_cost + c
                    best_path = first + reversedend

            # add node to explored set
            graph.explored_nodes.add(end_node)
            end_explored.append(end_node)
            for neighbor in graph[end_node]:
                new_path = None
                new_path = copy.deepcopy(end_path)
                new_path.append(neighbor)
                total_cost = end_cost + graph[end_node][neighbor]['weight']
                if neighbor not in end_explored and neighbor not in frontier_end:
                    frontier_end.append((total_cost,new_path))
                elif neighbor in frontier_end:
                    for c, p in frontier_end:
                        node = p[-1]
                        if neighbor == node:
                            if total_cost < c:
                                frontier_end.remove(neighbor)
                                frontier_end.append((total_cost,new_path))
                            break

    return best_path

# Exercise 2: Bidirectional A*
def bidirectional_a_star(graph, start, goal, heuristic=euclidean_dist_heuristic):
    frontier_end = PriorityQueue()
    frontier_start = PriorityQueue()
    frontier_start.append((0,[start]))
    frontier_end.append((0,[goal]))
    start_explored = []
    end_explored = []
    best_path = []
    best_cost = float("inf")

     # if frontier is empty return []
    if start == goal or not graph.has_node(start) or not graph.has_node(goal):
        return []

    while frontier_start and frontier_end:

        a_min = frontier_start.peek()[0]
        b_min = frontier_end.peek()[0]

        if ( a_min >= best_cost or frontier_start.isEmpty()) and (b_min >= best_cost or frontier_end.isEmpty()):
                    return best_path

        if a_min < best_cost:
            if not frontier_start.isEmpty():
                start_cost, start_path = frontier_start.pop()
                start_node = start_path[-1]
            if start_node == goal:
                curr_cost = get_path_cost(graph,start_path)
                if best_cost > start_cost:
                    best_cost = start_cost
                    best_path = start_path

            elif start_node in frontier_end:
                end_cost, end_path = frontier_end.get(start_node)
                reversedend = copy.deepcopy(end_path)
                reversedend.reverse()
                end_cost , curr_cost = get_path_cost(graph,end_path) , get_path_cost(graph,start_path)
                if best_cost > end_cost + start_cost:
                    best_cost = end_cost + start_cost
                    best_path = start_path[0:-1] + reversedend


             # add node to explored set
            start_explored.append(start_node)
            graph.explored_nodes.add(start_node)
            for neighbor in graph[start_node]:
                new_path = None
                new_path = copy.deepcopy(start_path)
                new_path.append(neighbor)
                total_cost = get_path_cost(graph,new_path)
                total_cost += heuristic(graph,neighbor,goal)
                if neighbor not in start_explored and neighbor not in frontier_start:
                    frontier_start.append((total_cost,new_path))
                elif neighbor in frontier_start:
                    for c, p in frontier_start:
                        node = p[-1]
                        if neighbor == node:
                            if total_cost < c:
                                frontier_start.remove(neighbor)
                                frontier_start.append((total_cost,new_path))
                            break
        if b_min < best_cost:
            if not frontier_end.isEmpty():
                end_cost, end_path = frontier_end.pop()
                end_node = end_path[-1]
                reversedend = copy.deepcopy(end_path)
                reversedend.reverse()
            if end_node == start:
                end_cost = get_path_cost(graph,end_path)
                if best_cost > end_cost:
                    best_cost = end_cost
                    best_path = reversedend

            elif end_node in frontier_start:
                c, start_path = frontier_start.remove(end_node)
                first = start_path[0:-1]
                reversedend = copy.deepcopy(end_path)
                reversedend.reverse()
                end_cost , c = get_path_cost(graph,end_path) , get_path_cost(graph,start_path)
                if best_cost > end_cost+c:
                    best_cost = end_cost + c
                    best_path = first + reversedend

            # add node to explored set
            graph.explored_nodes.add(end_node)
            end_explored.append(end_node)
            for neighbor in graph[end_node]:
                new_path = None
                new_path = copy.deepcopy(end_path)
                new_path.append(neighbor)
                total_cost = get_path_cost(graph,new_path)
                total_cost += heuristic(graph,neighbor,start)
                if neighbor not in end_explored and neighbor not in frontier_end:
                    frontier_end.append((total_cost,new_path))
                elif neighbor in frontier_end:
                    for c, p in frontier_end:
                        node = p[-1]
                        if neighbor == node:
                            if total_cost < c:
                                frontier_end.remove(neighbor)
                                frontier_end.append((total_cost,new_path))
                            break

    return best_path

# Exercise 3: Tridirectional UCS Search
def tridirectional_search(graph, goals):

    # check for duplicates in goals
    if len(goals) != len(set(goals)):
        return []

    start, mid, goal = goals[0], goals[1], goals[2]

    # print "start, mid, goal = ", goals[0], " , " , goals[1], " , ", goals[2]

    frontier_a = PriorityQueue() # a->b
    frontier_b = PriorityQueue() # b->c
    frontier_c = PriorityQueue() # c->a

    frontier_a.append((0,[start]))
    frontier_b.append(((0,[mid])))
    frontier_c.append((0,[goal]))

    a_explored = []
    c_explored = []
    b_explored = []

    min_ab = []
    cost_ab = float("inf")
    min_ca = []
    cost_ca = float("inf")
    min_bc = []
    cost_bc = float("inf")


    def min_path():

        print cost_ab , " " , min_ab
        print cost_bc, " " , min_bc
        print cost_ca, " ", min_ca
        ab = set(copy.deepcopy(min_ab))
        bc = set(copy.deepcopy(min_bc))
        ca = set(copy.deepcopy(min_ca))

        if cost_ab + cost_ca < cost_ab +cost_bc and cost_ab + cost_ca < cost_bc + cost_ca:
            if ca.issubset(ab):
                return min_ab
            elif ab.issubset(ca):
                return min_ca
            return min_ca + min_ab[1::]
        elif cost_ab + cost_bc < cost_bc + cost_ca:
            if ab.issubset(bc):
                return min_bc
            elif bc.issubset(ab):
                return min_ab
            return min_ab + min_bc[1::]
        else:
            if ca.issubset(bc):
                return min_bc
            elif bc.issubset(ca):
                return min_ca
            return min_bc + min_ca[1::]

    # if frontier is empty return []
    if not graph.has_node(start) or not graph.has_node(mid) or not graph.has_node(goal):
        return []

    while not frontier_b.isEmpty() or not frontier_a.isEmpty() or not frontier_c.isEmpty():
        a_min = frontier_a.peek()[0]
        b_min = frontier_b.peek()[0]
        c_min = frontier_c.peek()[0]

        if ( a_min >= cost_ab or frontier_a.isEmpty()) and (b_min >= cost_bc or frontier_b.isEmpty()) and (c_min  >= cost_ca or frontier_c.isEmpty()):
                    return min_path()

        if a_min < cost_ab: #     expand A
            if not frontier_a.isEmpty():
                curr_cost, path = frontier_a.pop()
                node = path[-1]
                if node == mid:
                    if cost_ab > curr_cost:
                        cost_ab = curr_cost
                        min_ab = path
                elif node in frontier_b:
                    end_cost, end_path = frontier_b.get(node)
                    reversedend = copy.deepcopy(end_path)
                    reversedend.reverse()
                    if cost_ab > end_cost + curr_cost:
                        cost_ab = end_cost + curr_cost
                        min_ab = path[0:-1] + reversedend
                elif node in frontier_c:
                    end_cost, end_path = frontier_c.get(node)
                    reversedend = copy.deepcopy(end_path)
                    reversedend.reverse()
                    if cost_ca > end_cost + curr_cost:
                        cost_ca = end_cost + curr_cost
                        min_ca = path[0:-1] + reversedend
                        if min_ca[-1] == goal:
                            min_ca.reverse()

                 # add node to explored set
                a_explored.append(node)
                graph.explored_nodes.add(node)
                for neighbor in graph[node]:
                    new_path = None
                    new_path = copy.deepcopy(path)
                    new_path.append(neighbor)
                    total_cost = curr_cost + graph[node][neighbor]['weight']
                    if neighbor not in a_explored and neighbor not in frontier_a:
                        frontier_a.append((total_cost,new_path))
                    elif neighbor in frontier_a:
                        for c, p in frontier_a:
                            n = p[-1]
                            if neighbor == n:
                                if total_cost < c:
                                    frontier_a.remove(neighbor)
                                    frontier_a.append((total_cost,new_path))
                                break

        if b_min < cost_bc:
            #         expand B
            if not frontier_b.isEmpty():
                curr_cost, path = frontier_b.pop()
                node = path[-1]
                if node == goal:
                    if cost_bc > curr_cost:
                        cost_bc = curr_cost
                        min_bc = path
                elif node in frontier_a:
                    end_cost, end_path = frontier_a.get(node)
                    reversedend = copy.deepcopy(end_path)
                    reversedend.reverse()
                    if cost_ab > end_cost + curr_cost:
                        cost_ab = end_cost + curr_cost
                        min_ab = path[0:-1] + reversedend
                        if min_ab[-1] == start:
                            min_ab.reverse()
                elif node in frontier_c:
                    end_cost, end_path = frontier_c.get(node)
                    reversedend = copy.deepcopy(end_path)
                    reversedend.reverse()
                    if cost_bc > end_cost + curr_cost:
                        cost_bc = end_cost + curr_cost
                        min_bc = path[0:-1] + reversedend

                 # add node to explored set
                b_explored.append(node)
                graph.explored_nodes.add(node)
                for neighbor in graph[node]:
                    new_path = None
                    new_path = copy.deepcopy(path)
                    new_path.append(neighbor)
                    total_cost = curr_cost + graph[node][neighbor]['weight']
                    if neighbor not in b_explored and neighbor not in frontier_b:
                        frontier_b.append((total_cost,new_path))
                    elif neighbor in frontier_b:
                        for c, p in frontier_b:
                            n = p[-1]
                            if neighbor == n:
                                if total_cost < c:
                                    frontier_b.remove(neighbor)
                                    frontier_b.append((total_cost,new_path))
                                break
        if c_min < cost_ca:
            #         expand C
            if not frontier_c.isEmpty():
                curr_cost, path = frontier_c.pop()
                node = path[-1]
                if node == start:
                    if cost_ca > curr_cost:
                        cost_ca = curr_cost
                        min_ca = path
                elif node in frontier_a:
                    end_cost, end_path = frontier_a.get(node)
                    reversed = copy.deepcopy(end_path)
                    reversed.reverse()
                    if cost_ca > end_cost + curr_cost:
                        cost_ca = end_cost + curr_cost
                        min_ca = path[0:-1] + reversed
                elif node in frontier_b:
                    end_cost, end_path = frontier_b.get(node)
                    reversed = copy.deepcopy(end_path)
                    reversed.reverse()
                    if cost_bc > end_cost + curr_cost:
                        cost_bc = end_cost + curr_cost
                        min_bc = path[0:-1] + reversed

                        if min_bc[-1] == mid:
                            min_bc.reverse()

                 # add node to explored set
                c_explored.append(node)
                graph.explored_nodes.add(node)
                for neighbor in graph[node]:
                    new_path = None
                    new_path = copy.deepcopy(path)
                    new_path.append(neighbor)
                    total_cost = curr_cost + graph[node][neighbor]['weight']
                    if neighbor not in c_explored and neighbor not in frontier_c:
                        frontier_c.append((total_cost,new_path))
                    elif neighbor in frontier_c:
                        for c, p in frontier_c:
                            n = p[-1]
                            if neighbor == n:
                                if total_cost < c:
                                    frontier_c.remove(neighbor)
                                    frontier_c.append((total_cost,new_path))
                                break

    return min_path()

# Exercise 4: Present an improvement on tridirectional search in terms of nodes explored
def tridirectional_upgraded(graph, goals, heuristic=euclidean_dist_heuristic):

    # check for duplicates in goals
    if len(goals) != len(set(goals)):
        return []

    start, mid, goal = goals[0], goals[1], goals[2]

    # print "start, mid, goal = ", goals[0], " , " , goals[1], " , ", goals[2]

    frontier_a = PriorityQueue() # a->b
    frontier_b = PriorityQueue() # b->c
    frontier_c = PriorityQueue() # c->a

    frontier_a.append((0,[start]))
    frontier_b.append(((0,[mid])))
    frontier_c.append((0,[goal]))

    a_explored = []
    c_explored = []
    b_explored = []

    min_ab = []
    cost_ab = float("inf")
    min_ca = []
    cost_ca = float("inf")
    min_bc = []
    cost_bc = float("inf")



    def min_path():

        print cost_ab , " " , min_ab
        print cost_bc, " " , min_bc
        print cost_ca, " ", min_ca
        ab = set(copy.deepcopy(min_ab))
        bc = set(copy.deepcopy(min_bc))
        ca = set(copy.deepcopy(min_ca))

        if cost_ab + cost_ca < cost_ab +cost_bc and cost_ab + cost_ca < cost_bc + cost_ca:
            if ca.issubset(ab):
                return min_ab
            elif ab.issubset(ca):
                return min_ca
            return min_ca + min_ab[1::]
        elif cost_ab + cost_bc < cost_bc + cost_ca:
            if ab.issubset(bc):
                return min_bc
            elif bc.issubset(ab):
                return min_ab
            return min_ab + min_bc[1::]
        else:
            if ca.issubset(bc):
                return min_bc
            elif bc.issubset(ca):
                return min_ca
            return min_bc + min_ca[1::]

    # if frontier is empty return []
    if not graph.has_node(start) or not graph.has_node(mid) or not graph.has_node(goal):
        return []

    while not frontier_b.isEmpty() or not frontier_a.isEmpty() or not frontier_c.isEmpty():
        a_min = frontier_a.peek()[0]
        b_min = frontier_b.peek()[0]
        c_min = frontier_c.peek()[0]

        if ( a_min >= cost_ab or frontier_a.isEmpty()) and (b_min >= cost_bc or frontier_b.isEmpty()) and (c_min  >= cost_ca or frontier_c.isEmpty()):
                    return min_path()

        if a_min < cost_ab: #     expand A
            if not frontier_a.isEmpty():
                curr_cost, path = frontier_a.pop()
                node = path[-1]
                if node == mid:
                    curr_cost = get_path_cost(graph,path)
                    if cost_ab > curr_cost:
                        cost_ab = curr_cost
                        min_ab = path
                elif node in frontier_b:
                    end_cost, end_path = frontier_b.get(node)
                    reversedend = copy.deepcopy(end_path)
                    end_cost , curr_cost = get_path_cost(graph,end_path) , get_path_cost(graph,path)
                    reversedend.reverse()
                    if cost_ab > end_cost + curr_cost:
                        cost_ab = end_cost + curr_cost
                        min_ab = path[0:-1] + reversedend
                elif node in frontier_c:
                    end_cost, end_path = frontier_c.get(node)
                    reversedend = copy.deepcopy(end_path)
                    reversedend.reverse()
                    end_cost , curr_cost = get_path_cost(graph,end_path) , get_path_cost(graph,path)
                    if cost_ca > end_cost + curr_cost:
                        cost_ca = end_cost + curr_cost
                        min_ca = path[0:-1] + reversedend
                        if min_ca[-1] == goal:
                            min_ca.reverse()

                 # add node to explored set
                a_explored.append(node)
                graph.explored_nodes.add(node)
                for neighbor in graph[node]:
                    new_path = None
                    new_path = copy.deepcopy(path)
                    new_path.append(neighbor)
                    total_cost = get_path_cost(graph,new_path)
                    total_cost += heuristic(graph,neighbor,mid)
                    if neighbor not in a_explored and neighbor not in frontier_a:
                        frontier_a.append((total_cost,new_path))
                    elif neighbor in frontier_a:
                        for c, p in frontier_a:
                            n = p[-1]
                            if neighbor == n:
                                if total_cost < c:
                                    frontier_a.remove(neighbor)
                                    frontier_a.append((total_cost,new_path))
                                break

        if b_min < cost_bc:
            #         expand B
            if not frontier_b.isEmpty():
                curr_cost, path = frontier_b.pop()
                node = path[-1]
                if node == goal:
                    curr_cost = get_path_cost(graph,path)
                    if cost_bc > curr_cost:
                        cost_bc = curr_cost
                        min_bc = path
                elif node in frontier_a:
                    end_cost, end_path = frontier_a.get(node)
                    reversedend = copy.deepcopy(end_path)
                    reversedend.reverse()
                    end_cost , curr_cost = get_path_cost(graph,end_path) , get_path_cost(graph,path)
                    if cost_ab > end_cost + curr_cost:
                        cost_ab = end_cost + curr_cost
                        min_ab = path[0:-1] + reversedend
                        if min_ab[-1] == start:
                            min_ab.reverse()
                elif node in frontier_c:
                    end_cost, end_path = frontier_c.get(node)
                    reversedend = copy.deepcopy(end_path)
                    reversedend.reverse()
                    end_cost , curr_cost = get_path_cost(graph,end_path) , get_path_cost(graph,path)
                    if cost_bc > end_cost + curr_cost:
                        cost_bc = end_cost + curr_cost
                        min_bc = path[0:-1] + reversedend

                 # add node to explored set
                b_explored.append(node)
                graph.explored_nodes.add(node)
                for neighbor in graph[node]:
                    new_path = None
                    new_path = copy.deepcopy(path)
                    new_path.append(neighbor)
                    total_cost = get_path_cost(graph,new_path)
                    total_cost += heuristic(graph,neighbor,goal)
                    if neighbor not in b_explored and neighbor not in frontier_b:
                        frontier_b.append((total_cost,new_path))
                    elif neighbor in frontier_b:
                        for c, p in frontier_b:
                            n = p[-1]
                            if neighbor == n:
                                if total_cost < c:
                                    frontier_b.remove(neighbor)
                                    frontier_b.append((total_cost,new_path))
                                break
        if c_min < cost_ca:
            #         expand C
            if not frontier_c.isEmpty():
                curr_cost, path = frontier_c.pop()
                node = path[-1]
                if node == start:
                    curr_cost = get_path_cost(graph,path)
                    if cost_ca > curr_cost:
                        cost_ca = curr_cost
                        min_ca = path
                elif node in frontier_a:
                    end_cost, end_path = frontier_a.get(node)
                    reversed = copy.deepcopy(end_path)
                    reversed.reverse()
                    end_cost , curr_cost = get_path_cost(graph,end_path) , get_path_cost(graph,path)
                    if cost_ca > end_cost + curr_cost:
                        cost_ca = end_cost + curr_cost
                        min_ca = path[0:-1] + reversed
                elif node in frontier_b:
                    end_cost, end_path = frontier_b.get(node)
                    reversed = copy.deepcopy(end_path)
                    reversed.reverse()
                    end_cost , curr_cost = get_path_cost(graph,end_path) , get_path_cost(graph,path)
                    if cost_bc > end_cost + curr_cost:
                        cost_bc = end_cost + curr_cost
                        min_bc = path[0:-1] + reversed

                        if min_bc[-1] == mid:
                            min_bc.reverse()

                 # add node to explored set
                c_explored.append(node)
                graph.explored_nodes.add(node)
                for neighbor in graph[node]:
                    new_path = None
                    new_path = copy.deepcopy(path)
                    new_path.append(neighbor)
                    total_cost = get_path_cost(graph,new_path)
                    total_cost += heuristic(graph,neighbor,start)
                    if neighbor not in c_explored and neighbor not in frontier_c:
                        frontier_c.append((total_cost,new_path))
                    elif neighbor in frontier_c:
                        for c, p in frontier_c:
                            n = p[-1]
                            if neighbor == n:
                                if total_cost < c:
                                    frontier_c.remove(neighbor)
                                    frontier_c.append((total_cost,new_path))
                                break

    return min_path()

# Extra Credit: Your best search method for the race
def custom_search(graph, goals):
    raise NotImplementedError