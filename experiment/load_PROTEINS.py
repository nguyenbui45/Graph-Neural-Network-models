from torch_geometric.datasets import TUDataset

dataset = TUDataset(root='', name = 'PROTEINS')
print(type(dataset))

