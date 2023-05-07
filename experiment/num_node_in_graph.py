import networkx as nx
import numpy as np
import torch
import random
from collections import Counter

PATH = '/home/nguyenbui45/Code_Directory/Graph/PROTEINS/'
FILE = {'edges':'PROTEINS_A.txt',
        'graph_indicator':'PROTEINS_graph_indicator.txt',
        'graph_labels':'PROTEINS_graph_labels.txt',
        'node_attr':'PROTEINS_node_attributes.txt',
        'node_label':'PROTEIN_node_labels.txt'}
NUM_GRAPH = 1113
NUM_NODE = 43471
NUM_EDGE = 162088

class GraphList(object):
    def __init__(self, g, graph_label, node_tags = None, node_features =  None):
        self.g = g
        self.graph_label = graph_label
        self.node_tags = node_tags
        self.neighbors = [] 
        self.edge_mat = 0
        self.max_neighbors = 0


        
def count_node():
    node_arr = []
    with open(PATH + FILE['graph_indicator']) as nodes:
        for i in range(NUM_NODE):
            node_arr.append(int(nodes.readline()))
    #print(node_arr)
    counter = Counter(node_arr)
    counter_str = sorted(counter.items())
    file = open("PROTEINS_nodes_in_graph.txt","w")
    for i in counter_str:
        file.write(str(i[1]))
        file.write('\n')
    file.close()





count_node()
        

