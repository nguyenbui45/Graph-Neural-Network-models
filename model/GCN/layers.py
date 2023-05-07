import torch
import torch.nn as nn
import math
from torch.nn.parameter import Parameter

class GraphConvolution(torch.nn.modules.module.Module):
    def __init__(self,in_features,out_features,bias=True):
        super(GraphConvolution,self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.cuda.FloatTensor(in_features,out_features)) # Create parameter of weight with shape (in_feaure,out_features)
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias',None) # Register a bias parameter in module

        self.reset_parameters()


    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1)) # calculate standard devidation of weight
        self.weight.data.uniform_(-stdv,stdv) # initilize weight value due to std. torch.nn.init.uniform_(tensor, a=0.0, b=1.0)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv,stdv)

    def forward(self,x,adj):
        support = torch.mm(x,self.weight) # matrix multiplication of input and weight
        output = torch.spmm(adj,support)  # sparse matrix multiplication of adj and support
        if self.bias is not None:
            return output + self.bias 
        else:
            return output


    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.in_features) + '->' + str(self.out_features) + ')'



#a = torch.cuda.FloatTensor(1,2)


