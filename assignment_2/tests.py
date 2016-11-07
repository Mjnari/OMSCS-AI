from __future__ import division
import random
import matplotlib.pyplot as plt
import pickle
import sys
sys.path.append('lib')
import osm2networkx
from search_submission import *
from visualize_graph import plot_search
from lib.networkx.classes import Graph

def testPQ():
    pq = PriorityQueue()
    temp_list = []

    for i in range(10):
        a = random.randint(0,10000)
        pq.append((a,'a'))
        temp_list.append(a)

    temp_list = sorted(temp_list)

    for i in temp_list:
        j = pq.pop()
        if not i == j[0]:
            return False

    return True

def draw_graph(graph, node_positions={}, start=None, goal=None, path=[]):

    explored = list(graph.get_explored_nodes())

    labels ={}
    for node in graph:
        labels[node]=node

    if not node_positions:
        node_positions = networkx.spring_layout(graph)

    networkx.draw_networkx_nodes(graph, node_positions)
    networkx.draw_networkx_edge_labels(graph,node_positions)
    networkx.draw_networkx_edges(graph, node_positions, style='dashed')
    networkx.draw_networkx_labels(graph,node_positions, labels)

    networkx.draw_networkx_nodes(graph, node_positions, nodelist=explored, node_color='g')

    if path:
        edges = [(path[i], path[i+1]) for i in range(0, len(path)-1)]
        networkx.draw_networkx_edges(graph, node_positions, edgelist=edges, edge_color='b')

    if start:
        networkx.draw_networkx_nodes(graph, node_positions, nodelist=[start], node_color='b')

    if goal:
        networkx.draw_networkx_nodes(graph, node_positions, nodelist=[goal], node_color='y')

    plt.plot()
    plt.show()

def testBFS(romania):
    start = 'a'
    goal = 'u'
    node_positions = {n: romania.node[n]['pos'] for n in romania.node.keys()}
    # draw_graph(romania,node_positions)
    romania.reset_search()
    path = breadth_first_search(romania, start, goal)
    print path, " nodes explored: ", romania.get_explored_nodes()
    return path == ['a', 's', 'f', 'b', 'u']

def test_path_cost(romania):
    node_positions = {n: romania.node[n]['pos'] for n in romania.node.keys()}
    # draw_graph(romania,node_positions)
    cost = get_path_cost(romania,['a', 's', 'f', 'b', 'u'])
    return cost == 535

def testCFS(romania):
    start = 'a'
    goal = 'u'
    node_positions = {n: romania.node[n]['pos'] for n in romania.node.keys()}
    romania.reset_search()
    path = uniform_cost_search(romania, start, goal)
    print romania.get_explored_nodes()
    draw_graph(romania,node_positions,start, goal,path)
    return path == [ 'a', 's', 'r', 'p', 'b', 'u']

def test_eucledian_dist(romania):
    dist = euclidean_dist_heuristic(romania,'p','u')

def test_a_star(romania):
    start = 'a'
    goal = 'u'
    node_positions = {n: romania.node[n]['pos'] for n in romania.node.keys()}
    romania.reset_search()
    path = a_star(romania, start, goal)
    print romania.get_explored_nodes()
    draw_graph(romania,node_positions,start, goal,path)
    return path == [ 'a', 's', 'r', 'p', 'b', 'u']

def test_bidirectional_loop(romania):
    for i in range(0,1000):
        snode = random.choice(romania.nodes())
        gnode = random.choice(romania.nodes())
        romania.reset_search()
        path = bidirectional_a_star(romania, snode,gnode)
        try:
            path_correct = networkx.shortest_path(romania, snode, gnode,'weight')
        except:
            print " No path possible"
            print "path: ", path
        # print "correct: ", path_correct
        # print "computed: " , path
        if path!=path_correct:
            print "Failed Search: ", snode , "->", gnode
            break

def test_bidirectional(romania):
    snode = "t"
    gnode = "v"
    romania.reset_search()
    path = bidirectional_a_star(romania, snode,gnode)
    try:
        path_correct = networkx.shortest_path(romania, snode, gnode,'weight')
    except:
        print " No path possible"
        print "path: ", path
    print "correct: ", path_correct
    print "computed: " , path
    if path!=path_correct:
        print "Failed Search: ", snode , "->", gnode
        print get_path_cost(romania,path_correct)
        print get_path_cost(romania,path)



def test_bidirectional2(romania):
    start = 't'
    goal = 'e'
    romania.reset_search()
    path = bidirectional_ucs(romania,start,goal)
    print "Found path: " , path
    path_correct = networkx.shortest_path(romania, start,goal,'weight')
    print "path: ", path_correct
    node_positions = {n: romania.node[n]['pos'] for n in romania.node.keys()}
    draw_graph(romania,node_positions,start,goal,path)
    return path == path_correct

def test_bidirectional_astar(romania):
    start = 'u'
    goal = 't'
    romania.reset_search()
    path = bidirectional_a_star(romania, start,goal)
    path_correct = networkx.shortest_path(romania, start, goal,'weight')
    print "correct: ", path_correct
    print "computed: " , path
    if path!=path_correct:
        print "Failed Search: ", start , "->", goal

def test_tridirectional(romania):
  print tridirectional_upgraded(romania, ['c','b','a'])


def romania_tri_test(romania):
    keys = romania.node.keys()
    triplets = [(x, y, z) for x in keys for y in keys for z in keys]

    for goals in triplets:
        print goals
        romania.reset_search()
        path = tridirectional_search(romania, goals)
        print " path = " , path


def main():

    romania = pickle.load(open('romania_graph.pickle', 'rb'))
    atl = pickle.load(open('atlanta_osm.pickle', 'rb'))
    atl.reset_search()
    # simple = Graph()
    # simple.add_node("a",{'pos':(100,100)})
    # simple.add_node("b",{'pos':(50,75)})
    # simple.add_node("c",{'pos':(100,50)})
    # simple.add_node("d",{'pos':(150,75)})
    # simple.add_edge('a','b',{'weight':2})
    # simple.add_edge('a','d',{'weight':10})
    # simple.add_edge('a','c',{'weight':50})
    # simple.add_edge('c','d',{'weight':7})
    # node_positions = {n: simple.node[n]['pos'] for n in simple.node.keys()}
    # draw_graph(simple,node_positions)
    # if test_path_cost(romania):
    #     print 'PASSED PC.'
    # if testPQ():
    #     print 'PASSED PQ.'
    # if testBFS(romania):
    #     print 'PASSED BFS.'
    # else:
    #     print 'FAILED BFS.'
    # if testCFS(romania):
    #     print 'PASSED CFS.'
    # if testCFS(romania):
    #     print 'PASSED CFS.'
    # if test_a_star(romania):
    #     print 'PASSED A*.'
    # if test_bidirectional2(atl):
    #     print 'PASSED BIDIRECTIONAL UCS.'
    if test_bidirectional_astar(romania):
        print 'PASSED BIDIRECTIONAL A*.'
    # if test_tridirectional(romania):
    #     print 'PASSED TRIDIRECTIONAL'
    # romania_tri_test(romania)

# then upload 'atlanta_search.json' to Gist
    # if test_bidirectional_astar(romania):
    #     print 'PASSED BIDIRECTIONAL A*'





if __name__ == '__main__':
  main()

