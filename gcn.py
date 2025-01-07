import torch
import torch.nn as nn
from torch.optim import Adam
from torch_geometric.nn import GATConv,GATv2Conv
from torch_geometric.utils import softmax


class GCN(nn.Module):
    """
    Basic Graph Convolutional Architecture. The default parameters correspond to the ones used in the mini-project report.
    """
    def __init__(self,
        in_features,
        n_hidden=32,
        num_classes,
        n_layers=2,
        n_layers_mlp=2,
        dropout=0.4,
        leaky_relu_slope=0.2):

        """ Initializes the GCN model.

        Args:
            in_features (int): number of input features per node.
            n_hidden (int): embedding size.
            num_classes (int): number of classes to predict for each node.
            n_layers (int) : number of GCNConv layers
            n_layers_mlp : number of GATConv layers
        """

        super(GCN, self).__init__()
        L=[]
        self.n_conv_layers=n_layers

        gcn1 = GCNConv(
            in_channels=in_features, out_channels=n_hidden)
        L.append(gcn1)

        for i in range (0,n_layers-1):
          L.append(GCNConv(
            in_channels=n_hidden, out_channels=n_hidden))

        for i in range (0,n_layers_mlp-1):
          L.append(nn.Linear(n_hidden,n_hidden))
        L.append(nn.Linear(n_hidden,num_classes))

        self.layers=nn.ModuleList(L)




    def forward(self, input_tensor: torch.Tensor , edge_index: torch.Tensor):
        """
        Performs a forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor representing node features.
            edge_index (torch.Tensor): Indexes of edges in the input graph.

        Returns:
            torch.Tensor: Output tensor after the forward pass.
        """
        x=input_tensor

        for i,layer in enumerate(self.layers):
          if i<len(self.layers)-1:
            if i<self.n_conv_layers:
              x=layer(x,edge_index)
            else:
              x=layer(x)
            x=F.relu(x)
          else:
            x=layer(x)

        return x 