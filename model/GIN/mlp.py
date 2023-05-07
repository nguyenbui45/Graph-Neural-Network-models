import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self,
            num_layers,
            input_dim,
            hidden_dim,
            output_dim
            ):

        super(MLP,self).__init__() 
        self.num_layers = num_layers
        self.linear_or_not = True

        if num_layers < 1:
            raise ValueError("Requires positive layer's number")
        elif num_layers == 1:
            self.linear = nn.Linear(input_dim, output_dim) 
        else:
            self.linear_or_not = False
            self.linears = nn.ModuleList() # Create a list of modules 
            self.batch_norms = nn.ModuleList() 

            self.linears.append(nn.Linear(input_dim,hidden_dim))
            for layer in range(num_layers -2):
                self.linears.append(nn.Linear(hidden_dim,hidden_dim))
            self.linears.append(nn.Linear(hidden_dim,output_dim))

            for layer in range(num_layers -1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))

    def forward(self,x):
        # linear 
        if self.linear_or_not: 
            return self.linear(x)
        else:
            # MLPs 
            h = x
            for layer in range(self.num_layers -1):
                h = F.relu(self.batch_norms[layer](self.linears[layer](h))) #inner MLP
            return self.linears[self.num_layers-1](h) # over another MLPs

