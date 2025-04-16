import torch
import torch_geometric
from torch.nn import Module, ModuleList, ModuleDict
from torch.nn import Parameter, ParameterList, ParameterDict
from torch.nn import BatchNorm1d, BatchNorm2d, BatchNorm3d
from torch.nn import LayerNorm
from torch.nn import Dropout, Dropout1d, Dropout2d, Dropout3d
from torch.nn import BCELoss, MSELoss, CrossEntropyLoss


# torch_geometric.transforms.add_positional_encoding

class Linear(Module):
    def __init__(self, out_dim, bias=True):
        super().__init__()
        self.linear = torch_geometric.nn.Linear(in_channels=-1,
                                                out_channels=out_dim,
                                                weight_initializer='kaiming_uniform',
                                                bias=bias,
                                                bias_initializer='zeros')
        self.linear.reset_parameters()

    def forward(self, x):
        return self.linear(x)


class Model(torch.nn.Module):
    def __init__(self, encoder=None, preditor=None):
        super().__init__()
        self.encoder = encoder
        self.preditor = preditor


class GNNConfig():
    def __init__(self):
        self.in_dim = -1
        self.hidden_dim = 128
        self.out_dim = 64
        self.num_layer = 2
        self.dropout_probability = 0.1
        self.jumping_knowledge_mode = None  # PyG: None "last" "cat" "max" "lstm"
        self.num_head = 8


class GCN(Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gnn = torch_geometric.nn.GCN(in_channels=config.in_dim,
                                          hidden_channels=config.hidden_dim,
                                          num_layers=config.num_layer,
                                          out_channels=config.out_dim,
                                          dropout=config.dropout_probability,
                                          jk=config.jumping_knowledge_mode)
        self.name = type(self).__name__

    def forward(self, node_embed, message_edge):
        return self.gnn.forward(x=node_embed, edge_index=message_edge)


class GAT(Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gnn = torch_geometric.nn.GAT(in_channels=config.in_dim,
                                          hidden_channels=config.hidden_dim,
                                          num_layers=config.num_layer,
                                          out_channels=config.out_dim,
                                          dropout=config.dropout_probability,
                                          jk=config.jumping_knowledge_mode,
                                          heads=config.num_head,
                                          v2=False)
        self.name = type(self).__name__

    def forward(self, node_embed, message_edge):
        return self.gnn.forward(x=node_embed, edge_index=message_edge)


class GATv2(Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gnn = torch_geometric.nn.GAT(in_channels=config.in_dim,
                                          hidden_channels=config.hidden_dim,
                                          num_layers=config.num_layer,
                                          out_channels=config.out_dim,
                                          dropout=config.dropout_probability,
                                          jk=config.jumping_knowledge_mode,
                                          heads=config.num_head,
                                          v2=True)
        self.name = type(self).__name__

    def forward(self, node_embed, message_edge):
        return self.gnn.forward(x=node_embed, edge_index=message_edge)


class GIN(Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gnn = torch_geometric.nn.GIN(in_channels=config.in_dim,
                                          hidden_channels=config.hidden_dim,
                                          num_layers=config.num_layer,
                                          out_channels=config.out_dim,
                                          dropout=config.dropout_probability,
                                          jk=config.jumping_knowledge_mode)
        self.name = type(self).__name__

    def forward(self, node_embed, message_edge):
        return self.gnn.forward(x=node_embed, edge_index=message_edge)


class GraphSAGE(Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gnn = torch_geometric.nn.GraphSAGE(in_channels=config.in_dim,
                                                hidden_channels=config.hidden_dim,
                                                num_layers=config.num_layer,
                                                out_channels=config.out_dim,
                                                dropout=config.dropout_probability,
                                                jk=config.jumping_knowledge_mode)
        self.name = type(self).__name__

    def forward(self, node_embed, message_edge):
        return self.gnn.forward(x=node_embed, edge_index=message_edge)


class PNA(Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gnn = torch_geometric.nn.PNA(in_channels=config.in_dim,
                                          hidden_channels=config.hidden_dim,
                                          num_layers=config.num_layer,
                                          out_channels=config.out_dim,
                                          dropout=config.dropout_probability,
                                          jk=config.jumping_knowledge_mode)
        self.name = type(self).__name__

    def forward(self, node_embed, message_edge):
        return self.gnn.forward(x=node_embed, edge_index=message_edge)



