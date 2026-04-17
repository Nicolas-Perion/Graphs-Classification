import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data
import networkx as nx

# Convert nx graphs to pytorch_geometric graphs
def nx_to_pyg(graph_list):
    """
    Nx graphs must be converted into exploitable graphs for pytorch_geometric.
    Such graphs are encoded into 3 elements :
      1) A tensor of the features of each node [N, features_amount]
      2) A tensor listing the starting and ending nodes of each edge in both directions (for undirected graphs) [2, 2*M]
      3) A tensor listing the attributes of each graph (in our case its label) [1]
    """
    pyg_data_list = []
    for G, target in graph_list:

        # Features of each node :
        degree = torch.tensor([G.degree(i) for i in range(G.number_of_nodes())], dtype = torch.float)
        clustering = torch.tensor([i for i in nx.clustering(G).values()], dtype = torch.float)
        closeness_centrality = torch.tensor([i for i in nx.closeness_centrality(G).values()], dtype = torch.float)
        betweenness_centrality = torch.tensor([i for i in nx.betweenness_centrality(G).values()], dtype = torch.float)
        x = torch.stack([degree, clustering, closeness_centrality, betweenness_centrality], dim = 1)

        # Encoding of edges in both directions
        edge_index = torch.tensor(list(G.edges), dtype = torch.long).t().contiguous()
        edge_index = torch.cat([edge_index, edge_index[[1, 0]]], dim = 1)

        # Attribute of the graph
        y = torch.tensor([target], dtype = torch.long)

        pyg_data_list.append(Data(x = x, edge_index = edge_index, y = y))
    return pyg_data_list

# Create GCN model
class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(4, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, 4)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        x = global_mean_pool(x, batch)

        x = F.dropout(x, p = 0.5, training = self.training)
        x = self.lin(x)
        return x

# Get GCN predictions
def get_gcn_predictions(model, loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for data in loader:
            out = model(data.x, data.edge_index, data.batch)
            pred = out.argmax(dim=1)
            all_preds.append(pred)
            all_labels.append(data.y)
    return torch.cat(all_labels).cpu().numpy(), torch.cat(all_preds).cpu().numpy()

# Compute feature importance (gradient : how much each feature influence the classification)
def gradient_feature_importance(model, loader):
    model.eval()
    feature_grads = []

    for data in loader:
        data.x.requires_grad_(True)
        out = model(data.x, data.edge_index, data.batch)
        
        pred_class = out.argmax(dim=1)
        score = out[range(len(pred_class)), pred_class].sum()
        score.backward()

        feature_grads.append(data.x.grad.abs().mean(dim=0))

    importance = torch.stack(feature_grads).mean(dim=0)
    feature_names = ["degree", "clustering", "closeness", "betweenness"]
    return dict(zip(feature_names, importance.tolist()))