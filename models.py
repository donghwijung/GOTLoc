import torch
import torch.nn as nn

from torch_geometric.nn.conv import MessagePassing, TransformerConv

from utils import make_cross_graph
        
class SimpleTConv(MessagePassing):
    def __init__(self, in_n, in_e, out_n, heads):
        super().__init__(aggr='add')
        self.TConv = TransformerConv(in_n, out_n, concat=False, heads=heads, dropout=0.5, edge_dim=in_e)
        self.act = nn.LeakyReLU()

    def forward(self, x, edge_index, edge_attr):
        x = self.TConv(x, edge_index, edge_attr)
        x = self.act(x)
        return x

class BigGNN(nn.Module):

    def __init__(self, N, heads):
        super().__init__()
        self.N = N
        in_n, in_e, out_n = 300, 300, 300
        self.TSALayers = nn.ModuleList([SimpleTConv(in_n, in_e, out_n, heads) for _ in range(N)])
        self.GSALayers = nn.ModuleList([SimpleTConv(in_n, in_e, out_n, heads) for _ in range(N)])
        self.TCALayers = nn.ModuleList([SimpleTConv(in_n, in_e, out_n, heads) for _ in range(N)])
        self.GCALayers = nn.ModuleList([SimpleTConv(in_n, in_e, out_n, heads) for _ in range(N)])

        self.SceneText_MLP = nn.Sequential(
            nn.Linear(300*2, 600), # TODO: input dimension is hardcoded now
            nn.LeakyReLU(),
            nn.Linear(600, 300),
            nn.LeakyReLU(),
            nn.Linear(300, 1),
            nn.Sigmoid()
        )

    def forward(self, x_1, x_2,
                      edge_idx_1, edge_idx_2,
                      edge_attr_1, edge_attr_2, device="cuda"):
        
        for i in range(self.N):
            ############# Self Attention #############
            x_1 = self.TSALayers[i](x_1, edge_idx_1, edge_attr_1)
            x_2 = self.GSALayers[i](x_2, edge_idx_2, edge_attr_2)
            ############# Self Attention #############
            # x_1_pooled = torch.mean(x_1, dim=0)
            # x_2_pooled = torch.mean(x_2, dim=0)
            # return x_1_pooled, x_2_pooled, torch.tensor(0.5)

            ############# Cross Attention #############
            len_x_1 = x_1.shape[0]
            len_x_2 = x_2.shape[0]
            edge_index_1_cross, edge_attr_1_cross = make_cross_graph(x_1.shape, x_2.shape) # First half of x_1_cross should be the original x_1
            edge_index_2_cross, edge_attr_2_cross = make_cross_graph(x_2.shape, x_1.shape) # First half of x_2_cross should be the original x_2
            x_1_cross = torch.cat((x_1, x_2), dim=0)
            x_2_cross = torch.cat((x_2, x_1), dim=0)
            x_1_cross = self.TCALayers[i](x_1_cross.to(device), edge_index_1_cross.to(device), edge_attr_1_cross.to(device))
            x_2_cross = self.GCALayers[i](x_2_cross.to(device), edge_index_2_cross.to(device), edge_attr_2_cross.to(device))
            x_1 = x_1_cross[:len_x_1] # TODO: Oh, this could be weird....... need to make sure the nodes and indices line up here
            x_2 = x_2_cross[:len_x_2]
            ############# Cross Attention #############
            # x_1 = F.normalize(x_1, p=2, dim=1)
            # x_2 = F.normalize(x_2, p=2, dim=1)
        
        # mean pooling
        x_1_pooled = torch.mean(x_1, dim=0)
        x_2_pooled = torch.mean(x_2, dim=0)

        # Concatenate and feed into SceneTextMLP
        x_concat = torch.cat((x_1_pooled, x_2_pooled), dim=0)
        out_matching = self.SceneText_MLP(x_concat)
        return x_1_pooled, x_2_pooled, out_matching

def permute_within_batch(x, batch):
    # Enumerate over unique batch indices
    unique_batches = torch.unique(batch)
    
    # Initialize list to store permuted indices
    permuted_indices = []

    for batch_index in unique_batches:
        # Extract indices for the current batch
        indices_in_batch = (batch == batch_index).nonzero().squeeze()
        
        # Permute indices within the current batch
        permuted_indices_in_batch = indices_in_batch[torch.randperm(len(indices_in_batch))]
        
        # Append permuted indices to the list
        permuted_indices.append(permuted_indices_in_batch)
    
    # Concatenate permuted indices into a single tensor
    permuted_indices = torch.cat(permuted_indices)

    return permuted_indices