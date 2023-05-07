import networkx as nx
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
import re 
from sklearn.model_selection import StratifiedKFold

PATH = '/home/nguyenbui45/Code_Directory/Graph/PROTEINS/'
FILE = {'edges':'PROTEINS_A.txt',
        'graph_indicator':'PROTEINS_graph_indicator.txt',
        'graph_labels':'PROTEINS_graph_labels.txt',
        'node_attr':'PROTEINS_node_attributes.txt',
        'node_label':'PROTEINS_node_labels.txt',
        'num_node':'PROTEINS_nodes_in_graph.txt'}
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
        self.node_features = node_features 
        self.max_neighbors = 0


def load_data(dataset,degree_as_tag=False):
    G_list = []
    graph_label_dict = {}
    line_read = 0 
    node_id = 0
    accumulate_node = 0
    num_node_arange = np.array([],dtype=np.int32)
    with open(PATH + FILE['graph_labels'] ,'r') as graph_label,open(PATH + FILE['graph_indicator'],'r') as node, open(PATH + FILE['node_attr']) as attr, open(PATH + FILE['node_label']) as node_tags, open(PATH + FILE['num_node'],'r') as num_node:
        for graph in range(NUM_GRAPH):
            g_label = int(graph_label.readline().strip('\n')) # check the label of the graph
            if not g_label in graph_label_dict:
                dummy_index = len(graph_label_dict)
                graph_label_dict[g_label] = dummy_index
            nodes_each_graph = int(num_node.readline().strip('\n'))
            accumulate_node = accumulate_node + nodes_each_graph  # check the number of nodes in each graph, the amount of each graph will be acumulative
            #print(accumulate_node,node_id)
            G = nx.Graph()
            G.add_nodes_from(np.arange(nodes_each_graph))
            num_node_arange = np.append(num_node_arange,np.arange(nodes_each_graph,dtype=np.int32))
            node_labels = []
            node_attribute = []
            # Add all nodes belong to the graph
            while(node_id != accumulate_node): 
                #check_belong_node = int(node.readline().strip('\n'))
                node_attr = float(attr.readline().strip('\n'))
                node_label = int(node_tags.readline().strip('\n')) # node_label short for node_label
                #G.add_node(node_id)
                node_attribute.append(node_attr)
                node_labels.append(node_label)
                node_id +=1 # after reading a new node, count it 
                # in case a new graph is read already, we need to add a node from previous loop.
                #if node_index < line_read: 
                #    G.add_node(node_index)
                #        node_attribute.append(node_attr)
                #        node_labels.append(no_label)
                #        node_index +=1

            #put the graph to GraphList object
            G_list.append(GraphList(G,g_label,node_labels,node_features = node_attribute))

    # Renaming nodes for each graph and create a dict to convert nodes 
    original_node_name = np.arange(NUM_NODE,dtype=np.int32) 
    NodeTransformDict = {original_node_name[i]:j for i,j in enumerate(num_node_arange)}

    # Add edges to the graph, create edge matrix, max_degree in GraphList class
    graph_id = 0
    with open(PATH + FILE['num_node'],'r') as num_node:
        max_node_id = int(num_node.readline().strip('\n'))
    with open(PATH +FILE['edges'],'r') as edge, open (PATH + FILE['num_node'],'r') as num_node:
        for i in range(NUM_EDGE):
            line = edge.readline().strip()
            line = re.split(r'\W',line)
            line.pop(1)
            #print(line)
            u,v = [int(j) for j in line]
            if u > max_node_id or v > max_node_id:
                if max_node_id == 42: # avoid max_node_id read line 1 again
                    dummy = int(num_node.readline().strip('\n'))
                max_node_id = max_node_id + int(num_node.readline().strip('\n'))
                graph_id += 1
                #print(max_node_id)
                G_list[graph_id].g.add_edge(NodeTransformDict[u-1],NodeTransformDict[v-1])

            else:
                G_list[graph_id].g.add_edge(NodeTransformDict[u-1],NodeTransformDict[v-1])



    # add max neighbors in 1 graph
    #for g in G_list:
    #    g.max_neighbors = max(g.g.degree())[1]
    for g in G_list:
        g.neighbors = [[] for i in range(len(g.g))]
        
        for i,j in g.g.edges():
            #add neighbor list in list
            g.neighbors[i].append(j)
            g.neighbors[j].append(i)
        degree_list = []
        for i in range(len(g.g)):
            g.neighbors[i] = g.neighbors[i] 
            degree_list.append(len(g.neighbors[i]))
        g.max_neighbors = max(degree_list)
        
        g.graph_label = graph_label_dict[g.graph_label]

        edges = [list(pair) for pair in g.g.edges()]
        edges.extend([[i,j] for j,i in edges])

        deg_list = list(dict(g.g.degree(range(len(g.g)))).values())
        g.edge_mat = torch.LongTensor(edges).transpose(0,1)

    # If treat degree as the label for the node
    if degree_as_tag:
        for g in G_list:
            g.node_tags = list(dict(g.g.degree).values())
    
    label_set = set([])
    # The reason when using set() inside node_labels is because it is the rule of thumb to group multiple of groups with (1) unchangeble elements, (2) don't allow duplicate. Thus we use set() to eliminate duplicate elements in the node labels list !
    for g in G_list:
        label_set = label_set.union(set(g.node_tags))
    label_set = list(label_set)
    index_as_label = {label_set[i]:i for i in range(len(label_set))}

    # Transform the node feature to one-hot encoding representation
    for g in G_list:
        g.node_features = torch.zeros(len(g.node_tags),len(label_set)) # [number of node labels [number of unique label] ]
        g.node_features[range(len(g.node_tags)), [index_as_label[label] for label in g.node_tags]] = 1  # Here the feature of node will be a vector of 0s and 1 corresponding to the unique label.
    

    return G_list,len(graph_label_dict)


def split_data(graph_list, seed, fold_idx):
    assert 0 <= fold_idx and fold_idx < 10, "fold_idx must be from 0 to 9."
    skf = StratifiedKFold(n_splits=10, shuffle = True, random_state = seed)

    labels = [graph.graph_label for graph in graph_list]
    idx_list = []
    for idx in skf.split(np.zeros(len(labels)), labels):
        idx_list.append(idx)
    train_idx, test_idx = idx_list[fold_idx]

    train_graph_list = [graph_list[i] for i in train_idx]
    test_graph_list = [graph_list[i] for i in test_idx]

    return train_graph_list, test_graph_list

        

                
    





if __name__ == "__main__":
    dataset = load_data("PROTEIN")
    print(dataset[0].edge_mat) 
    #nx.draw_networkx(dataset[0].g,with_labels=True)
    #plt.show()
    #print(dataset[0].node_features)
    #print(dataset[0].graph_label)
    #print(dataset[0].node_tags)
    #print(dataset[0].g.degree([0]))
    #print(dataset[0].max_neighbors)
