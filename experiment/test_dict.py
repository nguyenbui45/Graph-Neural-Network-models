import numpy as np

a = np.arange(100)
num_node = [40,40,20]
h=0
num_node_arange = np.array([])
for i in num_node:
    num_node_arange = np.append(num_node_arange,np.arange(i))
#num_node_arange = np.squeeze(num_node_arange,axis=0)

print(num_node_arange)

dict2index = {a[i]:j for i,j in enumerate(num_node_arange)}
print(dict2index)
