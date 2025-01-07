import torch
import torch.nn as nn
from torch.optim import Adam
from torch_geometric.nn import GATConv,GATv2Conv
from torch_geometric.utils import softmax

class GAT(torch.nn.Module):
    """
    Graph Attention Architecture. The default parameters correspond to the ones used in the mini-project report.
    """
    def __init__(self, in_features, n_hidden, heads,out_channels,n_layers,n_layers_mlp=2 ):
        """ Initializes the GCN model.

        Args:
            in_features (int): number of input features per node.
            n_hidden (int): embedding size.
            n_heads (int): number of attention heads in the Graph Attention Layers.
            num_classes (int): number of classes to predict for each node.
        """
        
        super().__init__()
        L=[]
        # Define the Graph Attention layers
        gat1 = GATConv(
            in_channels=in_features, out_channels=n_hidden,heads=heads)
        L.append(gat1)
        self.n_conv_layers=n_layers
        for i in range (0,n_layers-1):
          L.append(GATConv(
            in_channels=n_hidden*heads, out_channels=n_hidden,heads=heads))

        for i in range (0,n_layers_mlp-1):
          L.append(nn.Linear(n_hidden*heads,n_hidden*heads))
        L.append(nn.Linear(n_hidden*heads,out_channels))

        self.layers=nn.ModuleList(L)

    def forward(self, x, edge_index):
        """
        Performs a forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor representing node features.
            edge_index (torch.Tensor): Indexes of edges in the input graph.

        Returns:
            torch.Tensor: Output tensor after the forward pass.
        """
        for i,layer in enumerate(self.layers):
          if i<len(self.layers)-1:
            if i<self.n_conv_layers:
              x=layer(x,edge_index)
            else:
              x=layer(x)
            x=F.relu(x)
          else:
            x=layer(x)

        return x # Apply log softmax activation function