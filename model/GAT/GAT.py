import torch 
import torch.nn as nn
import time


time.time

class GAT(torch.nn.Module):
    def __init__(self, 
            num_of_layers,
            num_heads_per_layer,
            num_features_per_layer,
            add_skip_connection=True,
            bias=True,
            dropout_prob=0.6,
            log_attention_weights=False):
        super().__init__()

        num_heads_per_layer = [1] + num_heads_per_layer 
        gat_layers = [] # stores all gat layer 
        for i in range(num_of_layers):
            layer = GATBaseLayer(
                    num_in_features=num_features_per_layer[i] * num_heads_per_layer[i],
                    num_outfeatures = num_features_per_layer[i+1],
                    num_of_heads = num_heads_per_layer[i+1],
                    concat=True if i < num_of_layers -1 else False, # concat aggreation through neighbors except last operator does mean 
                    activation= nn.ELU() if i < num_of_layers - 1 else None,
                    dropout_prob = drop_prob,
                    add_skip_connection=add_skip_connection,
                    bias=bias,
                    log_attention_weight = log_attention_weights 
                    )
            gat_layers.append(layer) 

        self.gat_net = nn.Sequential(*gat_layers,)

    def forward(self,data):
        return self.gat_net(data) 


class GATBaseLayer(torch.nn.Module):
    '''
    Base class for GAT implementation
    '''
    head_dim = 1

    def __init__(self,
                num_in_features,
                num_out_features,
                num_of_heads,
                concat=True,
                activation=nn.ELU(),
                dropout_prob=0.6,
                add_skip_connection=True,
                bias=True,
                log_attention_weights=False
                ):
        super(self,GATBaseLayer).__init__() 

        self.num_heads = num_of_heads 
        self.num_out_features = num_out_features
        self.concat = concat
        self.add_skip_connection = add_skip_connection 

        # Declare trainable weight W 
        #nn.Linear() provide linear transformation to the data
        self.linear_proj = nn.Linear(num_in_features,num_of_heads*num_out_features,bias=False) 
        
        # self attention on node e_ij = a(Wh_i,Wh_j) (1) 
        # Instead of deriving exact equation, we can split a into 2 elements and dot product them correspending for each Wh_k. Remind that inside paratheis is a concatenation operation, outside them is dot product with a
        self.scoring_fn_target = nn.Parameter(torch.tensor(1,num_of_heads,num_out_features))
        self.scoring_fn_source = nn.Parameter(torch.Tensor(1,num_of_heads,num_out_features)) 

        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(num_of_heads * num_out_features)) 
        elif bias and not concat: 
            self.bias = nn.Parameter(torch.Tensor(num_out_features)) 
        else: 
            self.register_parameter('bias',None) 

        if add_skip_connection:
            self.skip_proj = nn.Linear(num_in_features,num_of_heads * num_out_features,bias=False) 
        else:
            self.register_parameter('skip_proj',None) 

        
        # declare activation & dropout
        self.leakyReLU = nn.LeakyReLU(0.2) 
        self.softmax = nn.Softmax(dim=-1) 
        self.activation = activation # nn.ELU() 
        self.dropout = nn.Dropout(p=dropout_prob) 

        self.log_attention_weights = log_attention_weights 
        self.attention_weights = None # weight for attetion scheme

        self.init_params() 

    def init_params(self): 
        ''' 
        method to initilize weight 
        '''
        nn.init.xavier_uniform_(self.linear_project.weight) 
        nn.init.xavier_uniform_(self.scoring_fn_target) 
        nn.init.xavier_uniform_(self.scoring_fn_source) 

    def skip_concat_bias(self,
            attention_coefficients,
            in_nodes_features,
            out_nodes_features):
        if self.log_attention_weights:
            self.attention_weights = attention_coefficients

        if self.add_skip_connection:
            if out_nodes_features.shape[-1] == in_nodes_features.shape[-1]:
                out_nodes_features += in_nodes_features.unsqueeze(1) 
            else:
                out_nodes_features += self.skip_proj(in_nodes_features).view(-1,self.num_of_heads,self.num_out_features) 


        if self.concat:
            out_nodes_features = out_nodes_features.view(-1,self.num_of_heads * self.num_out_features) 
        else: 
            out_nodes_features = out_nodes_features.mean(dim=self.head_dim) 

        if self.bias is not None:
            return out_nodes_features if self.activation is None else self.activation(out_nodes_features) 



class GATCoreLayer(GATBaseLayer):
    def __init__(self,
            num_in_features,
            num_out_features,
            num_of_heads,
            concat,
            activation,
            dropout_prob,
            add_skip_connection,
            bias,
            log_attention_weights):
        super().__init__(num_in_features,
                num_out_features,
                num_of_heads,
                concat,
                activation,
                dropout_prob,
                add_skip_connection,
                bias,
                log_attention_weights) 
        
    def forward(self,data):
        '''
        Linear Projection 
        '''

        in_nodes_features, edge_index = data # unpack data to node features  and edge_index
        num_of_nodes = 


