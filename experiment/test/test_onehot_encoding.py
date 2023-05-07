from tqdm import tqdm
import numpy as np 
import torch

#for i in range(200):
#    iteration = 0 
#    pbar = tqdm(range(100),unit='batch')
#    for pos in pbar:
#        iteration +=1
#        pbar.set_description('iteration %d' %(iteration))

#idx = np.arange(1000)
#for i in range(0,1000,64):
#    sample_idx = idx[i:i+64]
#    print(sample_idx)

#one_hot = torch.zeros(10,10)
#print(one_hot)

tagset = set([])
nodetag = []
nodetag.append(range(10))
nodetag.append(range(6))
nodetag.append(range(9))
nodetag.append([56])
for i in range(len(nodetag)):
    tagset = tagset.union(set(nodetag[i]))
tagset = list(tagset)
index2label = {tagset[i]: i for i in range(len(tagset))}
print(index2label)

feature = [[],[],[],[]]
for i in range(len(nodetag)):
    feature[i] = torch.zeros(len(nodetag[i]),len(tagset))
    feature[i][range(len(nodetag[i])),[index2label[label] for label in nodetag[i]]] = 1 

print(feature)
