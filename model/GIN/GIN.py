import torch
import torch.nn as nn
import torch.nn.functional as F
from model.GIN.mlp import MLP

class GIN(nn.Module):
    def __init__(self,
            num_layers,
            num_mlp_layers,
            input_dim,
            hidden_dim,
            output_dim,
            final_dropout,
            learn_eps,
            graph_pooling_type,
            neighbor_pooling_type,
            device):

        super(GIN,self).__init__()

        self.num_layers = num_layers
        self.eps = nn.Parameter(torch.zeros(self.num_layers -1 )) # epsilon parameter declaration = (0,0,0,0,..., num_layers - 1 )
        self.final_dropout = final_dropout # dropout ratio of final layers
        self.graph_pooling_type = graph_pooling_type # option for the graph aggregate function (sum/average)
        self.neighbor_pooling_type = neighbor_pooling_type # option for neighbor_pooling type (mean/max) 
        self.learn_eps = learn_eps # True:distinguish center node & neighboring node, False: aggregate center node & its neighboring nodes
        self.device = device
    
        self.mlps = nn.ModuleList() # declare list of MLP
        self.batch_norms = nn.ModuleList() # declate list of batch norm

        for layer in range(self.num_layers - 1):
            if layer ==0:
                self.mlps.append(MLP(num_mlp_layers,input_dim,hidden_dim,hidden_dim))
            else:
                self.mlps.append(MLP(num_mlp_layers,hidden_dim,hidden_dim,hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        

        self.linear_prediction = torch.nn.ModuleList() # Linear function to map hidden representation to prediction score
        for layer in range(self.num_layers):
            if layer == 0:
                self.linear_prediction.append(nn.Linear(input_dim,output_dim))
            else:
                self.linear_prediction.append(nn.Linear(hidden_dim,output_dim))



        '''
        Method's name: ___preprocess_neighbor_maxpool
        input:
            + batch_graph: a unit batch of graph to take.
        output:
            + A tensor of neighbor list after pooling

        function:
            + Create a 'stx_idx' variable equal (number of nodes in the graph + enumearation).Ex: graph 1 has 4 nodes => 4 + 0 (its order in the list) = 4, graph 2 as 2 nodes => 2 + 1 (its order in list) = 3 
            + Traversal every node in the graph:
                (1) add the neighbors indices with an stx_idx offset - store it to 'pad' variable 
                (2) extend the slot in 'pad' variable to a number equal to (max_degree - len(pad)), values in extended slot are equal -1 
                (3) check if eps parameters learning option, if not then append a value (len(neighbors) + stx_idx(i)
         '''

    def __preprocess_neighbor_maxpool(self,batch_graph):
        max_degree = max([graph.max_neighbors for graph in batch_graph])
        stx_index = [0]
        padded_neighbor_list = []
        for i,graph in enumerate(batch_graph):
            stx_idx.append(stx_index[i] + len(graph.g))
            padded_neighbors = []
            for j in range(len(graph.neighbors)): # traversal over the neighbor list (w.r.t every node) in a graph. j: node 1, node 2, node 3 ..
                pad = [stx_index[i] + n for n in graph.neighbors[j]] # add the node's neighbor list with stx_index[i] of the graph
                pad.extend([-1] * (max_degree - len(pad)))
                if not self.learn_esp:
                    pad.append( j + stx_index[i]) # if not learn esp, add the center node to the padding list
                padded_neighbors.append(pad) #  add the node's level padded neighbors list to graph's level padded neighbor list

            padded_neighbor_list.extend(padded_neighbors) # add the graph's neighbor list to the batch's padded neighbor list

        return torch.LongTensor(padded_neighbor_list)


    '''
    Method's name: __preprocess_neighbor_sumavepool
    input: 
        + batch_graph: a unit batch of graph to take
    output:
        + Create sparse matrix
    function:
        + Process with the edge matrix list 
            (1) Append the edge matrix list by stx_index (cumulative) 
            (2) Transform the edge matrix list to a vector 
            (3) Create a ones matrix with (2)shape 

        + Construct a sparse matrix 
            (1) if enable no distinguish center node and its neighbors, add a square matrix self-loop of center node to adjacency matrix.
            (2) Declare a vector of 1s represent for N self-loop (N is the no. nodes)
            (3) Concat adjacency_index with the self-loop index
            (4) Concat the adjacency_ele with (2) 
            (5) Return a sparse matrix representation with adjacency blocks

    '''
    def __preprocess_neighbor_sumavepool(self,batch_graph):
        edge_mat_list = []
        stx_index = [0]

        for i,graph in enumerate(batch_graph):
            stx_index.append(stx_index[i] + len(graph.g))
            edge_mat_list.append(graph.edge_mat + stx_index[i])
        Adj_block_index  = torch.concat(edge_mat_list,1)
        Adj_block_element = torch.ones(Adj_block_index.shape[1])

        if not self.learn_eps: # if not distinguish the center node & its neighbors,  add selfloop to the center node
            num_nodes = stx_index[-1]
            self.self_loop = torch.LongTensor([range(num_nodes),range(num_nodes)])
            element = torch.ones(num_nodes)
            Adj_block_index = torch.concat([Adj_block_index,self.self_loop],1)
            Adj_block_element = torch.concat([Adj_block_element,element],0)

        Adj_block = torch.sparse.FloatTensor(Adj_block_index,Adj_block_element,torch.Size([stx_index[-1],stx_index[-1]]))

        return Adj_block.to(self.device)

    '''
    Method name: __preprocess_graphpool
    input:
        + batch graph: a unit of batch

    output:
        + A sparse matrix of graph pooling

    function:
        + Add cumulative nodes stx_index
        + For either graph_pooling_type == "average" or == "summation",extend the element pooling
        + Extend index list according to pair [graph,nodes_index]
        + Construct the sparse matrix of index & element after pooling.

    '''

    def __preprocess_graphpool(self,batch_graph):
        stx_index = [0]
        for i,graph in enumerate(batch_graph):
            stx_index.append(stx_index[i] + len(graph.g))

        index  = []
        element = []

        for i,graph in enumerate(batch_graph):
            if self.graph_pooling_type == "average":
                element.extend([1. / len(graph.g)] * len(graph.g))

            else:
                element.extend([1] * len(graph.g))
            index.extend([[i,j] for j in range(stx_index[i],stx_index[i+1],1)])

        element = torch.FloatTensor(element)
        index = torch.LongTensor(index).transpose(0,1)
        graph_pool = torch.sparse.FloatTensor(index,element,torch.Size([len(batch_graph),stx_index[-1]]))

        return graph_pool.to(self.device)


    '''
    Method name:maxpool
    input:
        + h : feature vector
        + padded_neighbor_list:  list of padded neighbor after preprocessing
    output:
        + max pooling tensor
    function:
        + Find the min list in feature vector 
        + Concatenate the min list and the feature vector: h_with_min
        + Max pooling feature vector with padded_neighbor_list

    '''

    def maxpool(self,h,padded_neighbor_list):
        temp = torch.min(h,dim=0)[0]
        h_with_temp = torch.concat([h,temp.reshape((1,-1)).to(self.device)])
        pooled_rep = torch.max(h_with_temp[padded_neighbor_list],dim=1)[0]
        return pooled_rep 

    '''
    Method name: next_layer_esp 
    input: 
        + h: feature vector
        + layer: layer to perform 4.1 equation
        + padded_neighbor_list: list of padded neighbor
        + Adj_block: Adjacency matrix of edge
    output: Feature vector after applied 4.1 equation 

    function:
        + Check if max pooling is enable, if yes pass h and padded_neighbor_list to maxpool function
        + Else multiply feature vector and edge adjacency matrix
            (1) If average pooling is enable, find the average pooling. 
        + Applied 4.1 equation, batchnorm and relu activation 
    '''

    def next_layer_eps(self,h,layer,padded_neighbor_list=None,Adj_block=None):
        if self.neighbor_pooling_type == "max":
            pooled = self.maxpool(h,padded_neighbor_list)
        else:
            pooled = torch.spmm(Adj_block,h)
            if self.neighbor_pooling_type == "average":
                degree = torch.spmm(Adj_block,torch.ones((Adj_block.shape[0],1)).to(self.device))
                pooled = pooled/degree

        pooled = pooled + (1 + self.eps[layer]) * h
        pooled_rep = self.mlps[layer](pooled)
        h = self.batch_norm[layer](pooled_rep)
        h = F.relu(h)
        return h


    '''
    Method name: next_layer 
    input: 
        + h: feature vector
        + layer: layer to perform 4.1 equation
        + padded_neighbor_list: list of padded neighbor
        + Adj_block: Adjacency matrix of edge
    output: Feature vector after applied 4.1 equation 

    function:
        + Same as next_layer_eps, except we assume epsilon = 0, thus we don't learn epsilon, some line will be ignore w.r.t next_layer_eps
        + Check if max pooling is enable, if yes pass h and padded_neighbor_list to maxpool function
        + Else multiply feature vector and edge adjacency matrix
            (1) If average pooling is enable, find the average pooling. 
        + Applied 4.1 equation, batchnorm and relu activation 
    '''
    def next_layer(self,h,layer,padded_neighbor_list=None,Adj_block=None):
        if self.neighbor_pooling_type == "max":
            pooled = self.maxpool(h,padded_neighbor_list)
        else:
            pooled = torch.spmm(Adj_block,h)
            if self.neighbor_pooling_type == "average":
                degree = torch.spmm(Adj_block,torch.ones((Adj_block.shape[0],1)).to(self.device))
                pooled = pooled/degree

        # We don't learn espsilon here
        pooled_rep = self.mlps[layer](pooled)
        h = self.batch_norms[layer](pooled_rep)
        h = F.relu(h)
        return h

    '''
    Method name: forward 
    input: 
        + batch_graph: a unit of batch
    output:
        + score_over_layer: prediction over GIN layer

    function:
        + Forward GIN layer
        + Transform node feature into a vector
        + Perform graphpooling over the batch
        + Check neighbor_pooling_type
        + Determine hidden layers representation

    '''
    def forward(self,batch_graph):
        print("batch_graph",len(batch_graph))
        X_concat = torch.cat([graph.node_features for graph in batch_graph],0).to(self.device)
        print(X_concat.shape)
        graph_pool = self.__preprocess_graphpool(batch_graph)
        
        if self.neighbor_pooling_type == 'max':
            padded_neighbor_list = self.__preprocess_neighbor_maxpool(batch_graph)

        else:
            Adj_block = self.__preprocess_neighbor_sumavepool(batch_graph)

        hidden_rep = [X_concat]
        h = X_concat 
        
        for layer in range(self.num_layers - 1):
            if self.neighbor_pooling_type == 'max' and self.learn_eps:
                h = self.next_layer_eps(h,layer, padded_neighbor_list = padded_neighbor_list)
            elif self.neighbor_pooling_type == 'max' and not self.learn_eps:
                h =self.next_layer(h,layer,padded_neighbor_list = padded_neighbor_list)
            elif not self.neighbor_pooling_type == 'max' and self.learn_eps:
                h = self.next_layer_eps(h,layer,Adj_block = Adj_block)
            elif not self.neighbor_pooling_type == 'max' and not self.learn_eps:
                h = self.next_layer(h,layer,Adj_block = Adj_block)

            hidden_rep.append(h)

        score_over_layer = 0 
        
        # pooling over all nodes in each graph
        for layer,h in enumerate(hidden_rep):
            pooled_h = torch.spmm(graph_pool,h)
            print("pooled_h",pooled_h.shape)
            score_over_layer += F.dropout(self.linear_prediction[layer](pooled_h),self.final_dropout,training=self.training)
            print("score_over_layer",score_over_layer.shape)

        return score_over_layer

