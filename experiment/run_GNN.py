import sys 
import os
import argparse 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from collections import Counter 

#local import
sys.path.insert(1,os.path.join(os.path.dirname(__file__),'..'))
print(sys.path)
#from model.GCN import GCN
#from model.GIN import MLP
from model import GIN
#from model.GAT import GAT
from utils import load_data,split_data

def train(args,model,device,batch_graphs,optimizer,epoch,loss_function):
    model.train()
    #output = model(train_graphs)

    pbar = tqdm(range(args.iters_per_epoch),unit='batch')
    loss_accumulate = 0

    for percentage in pbar:
        selected_idx = np.random.permutation(len(batch_graphs))[:args.batch_size]
        batch_graphs = [batch_graphs[idx] for idx in selected_idx]
        output = model(batch_graphs)
        labels = torch.LongTensor([graph.graph_label for graph in batch_graphs]).to(device)
        print("output_shape",output.shape)
        print("labels_shape",labels.shape)
        loss =loss_function(output,labels)

        optimizer.zero_grad()
        loss.backward() 
        optimizer.step() 

        loss = loss.detach().cpu().numpy() 
        loss_accumulate += loss

        pbar.set_description('epoch: %d' % (epoch))
        
    
    average_loss = loss_accumulate/args.iters_per_epoch
    print("Training loss: %f"% (average_loss))

    return average_loss


def pass_data(model,graphs,minibatch_size=64):
    output = []
    idx = np.arange(len(graphs))
    for i in range(0,len(graphs),minibatch_size):
        sampled_idx = idx[i:i + minibatch_size] 
        if len(sampled_idx) == 0:
            continue
        output.append(model([graphs[j] for j in sampled_idx]).detach())
    return torch.cat(output,0)

def test(args,model,device, test_graphs, epoch):
    model.eval()

#    output = pass_data(model.train_graphs)
#    prediction = output.max(1,keepdim=True)[1]
#    labels = torch.LongTensor([graph.labels for graph in train_graphs]).to(device)
#    accuracy = prediction.eq(labels.view_as(prediction)).sum().cpu().items()
#    average_accuracy = accuracy / float(len(train_graphs))
    output = pass_data(model,test_graphs)
    test_predictions = output.max(1,keepdim=True)[1] 
    labels = torch.LongTensor([graph.graph_label for graph in test_graphs]).to(device)
    test_acc = test_predictions.eq(labels.view_as(test_predictions)).sum().cpu().item()
    ave_test_acc = test_acc / float(len(test_graphs))

    print("accuracy test: %f" % (ave_test_acc))

    return ave_test_acc



def main():
    # Create parser for argument
    parser = argparse.ArgumentParser(description= "argument for training GNNs")
    parser.add_argument('--dataset',type=str,default="PROTEIN")
    parser.add_argument('--device',type=int,default=0)
    parser.add_argument('--batch_size',type=int,default=32)
    parser.add_argument('--epochs',type=int,default=350)
    parser.add_argument('--lr',type=float,default=0.01)
    parser.add_argument('--degree_as_tags',action='store_true')
    parser.add_argument('--iters_per_epoch',type=int,default=50)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--fold_idx',type=int,default=0)
    parser.add_argument('--filename',type=str,default="training_history.txt")
    parser.add_argument('--learn_eps',action='store_true')

    # Model configuration
    parser.add_argument('--num_layers', type=int, default = 5, help='number of layers for layers INCLUDING the input one')
    parser.add_argument('--num_mlp_layers', type=int, default=2,help='number of layers for MLP EXCLUDING the input')
    parser.add_argument('--hidden_dims',type=int,default =64,help='number of hidden units')
    parser.add_argument('--final_dropout',type=float,default=0.5,help='final layer dropout')
    parser.add_argument('--graph_pooling_type',type=str,default='sum',choices=['sum','average'],help='pooling for over nodes in a graph')
    parser.add_argument('--neighbor_pooling_type',type=str, default="sum", choices=['sum','average','max'],help='pooling for over neighboring nodes')
    
    args = parser.parse_args()
    torch.manual_seed(0)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available:
        torch.cuda.manual_seed_all(0)

    dataset, num_classes = load_data(args.dataset,args.degree_as_tags)
    #print(dataset[0].node_features.shape)
    train_data, test_data = split_data(dataset,args.seed,args.fold_idx)
    X_concat = torch.cat([graph.node_features for graph in train_data],0)
    #print(X_concat.shape)
    #print(train_data[2].node_features)


    model = GIN.GIN(args.num_layers,args.num_mlp_layers,train_data[0].node_features.shape[1],args.hidden_dims,num_classes,args.final_dropout,args.learn_eps,args.graph_pooling_type,args.neighbor_pooling_type,device).to(device)

    #specify optimizers
    optimizer = optim.Adam(model.parameters(),lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50,gamma=0.5)
    loss_function = nn.CrossEntropyLoss() 

    for epoch in range(args.epochs + 1):
        scheduler.step() 
        ave_loss = train(args,model,device,train_data,optimizer,epoch,loss_function)
        ave_test_acc = test(args,model,device,test_data,epoch)

        with open(args.filename,'w') as f:
            f.write("%f %f" % (ave_loss,ave_test_acc))
            f.write('\n')
        print("")

        print(model.eps)




if __name__ == "__main__":
    main() 









