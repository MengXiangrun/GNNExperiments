import random
import numpy as np
import pandas as pd
import torch_geometric
import torch
from torch_geometric.datasets import Planetoid, CitationFull, Amazon, Coauthor


class SingleHomogeneousGraphNodeClassification(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.name = type(self).__name__
        self.node_feature = None
        self.node_label = None
        self.num_node_label = None

        self.edge_feature = None
        self.edge_label = None
        self.num_edge_label = None

        self.message_edge = None

        self.predict_node = None
        self.predict_edge = None

        self.graph_label = None
        self.num_graph_label = None

        self.count_label()

    def count_label(self):
        if self.node_label is not None:
            self.num_node_label = len(torch.unique(self.node_label))
        if self.edge_label is not None:
            self.num_edge_label = len(torch.unique(self.edge_label))


class Cora_Planetoid(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.name = type(self).__name__
        data = Planetoid(root=f'./{self.name}', name='Cora', split='full')[0]

        self.train_graph = SingleHomogeneousGraphNodeClassification()
        self.train_graph.message_edge = data.edge_index
        self.train_graph.node_feature = data.x
        self.train_graph.node_label = data.y
        self.train_graph.predict_node = torch.where(data.train_mask)[0]

        self.valid_graph = SingleHomogeneousGraphNodeClassification()
        self.valid_graph.message_edge = data.edge_index
        self.valid_graph.node_feature = data.x
        self.valid_graph.node_label = data.y
        self.valid_graph.predict_node = torch.where(data.val_mask)[0]

        self.test_graph = SingleHomogeneousGraphNodeClassification()
        self.test_graph.message_edge = data.edge_index
        self.test_graph.node_feature = data.x
        self.test_graph.node_label = data.y
        self.test_graph.predict_node = torch.where(data.test_mask)[0]

        self.train_graph.count_label()
        self.valid_graph.count_label()
        self.test_graph.count_label()

        print('Done')


class CiteSeer_Planetoid(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.name = type(self).__name__
        data = Planetoid(root=f'./{self.name}', name='CiteSeer', split='full')[0]

        self.train_graph = SingleHomogeneousGraphNodeClassification()
        self.train_graph.message_edge = data.edge_index
        self.train_graph.node_feature = data.x
        self.train_graph.node_label = data.y
        self.train_graph.predict_node = torch.where(data.train_mask)[0]

        self.valid_graph = SingleHomogeneousGraphNodeClassification()
        self.valid_graph.message_edge = data.edge_index
        self.valid_graph.node_feature = data.x
        self.valid_graph.node_label = data.y
        self.valid_graph.predict_node = torch.where(data.val_mask)[0]

        self.test_graph = SingleHomogeneousGraphNodeClassification()
        self.test_graph.message_edge = data.edge_index
        self.test_graph.node_feature = data.x
        self.test_graph.node_label = data.y
        self.test_graph.predict_node = torch.where(data.test_mask)[0]

        self.train_graph.count_label()
        self.valid_graph.count_label()
        self.test_graph.count_label()

        print('Done')


class PubMed_Planetoid(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.name = type(self).__name__
        data = Planetoid(root=f'./{self.name}', name='PubMed', split='full')[0]

        self.train_graph = SingleHomogeneousGraphNodeClassification()
        self.train_graph.message_edge = data.edge_index
        self.train_graph.node_feature = data.x
        self.train_graph.node_label = data.y
        self.train_graph.predict_node = torch.where(data.train_mask)[0]

        self.valid_graph = SingleHomogeneousGraphNodeClassification()
        self.valid_graph.message_edge = data.edge_index
        self.valid_graph.node_feature = data.x
        self.valid_graph.node_label = data.y
        self.valid_graph.predict_node = torch.where(data.val_mask)[0]

        self.test_graph = SingleHomogeneousGraphNodeClassification()
        self.test_graph.message_edge = data.edge_index
        self.test_graph.node_feature = data.x
        self.test_graph.node_label = data.y
        self.test_graph.predict_node = torch.where(data.test_mask)[0]

        self.train_graph.count_label()
        self.valid_graph.count_label()
        self.test_graph.count_label()
        print('Done')


class Cora_CitationFull(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.name = type(self).__name__
        data = CitationFull(root=f'./{self.name}', name='cora')[0]

        self.train_graph = SingleHomogeneousGraphNodeClassification()
        self.train_graph.message_edge = data.edge_index
        self.train_graph.node_feature = data.x
        self.train_graph.node_label = data.y
        self.train_graph.predict_node = torch.where(data.train_mask)[0]

        self.valid_graph = SingleHomogeneousGraphNodeClassification()
        self.valid_graph.message_edge = data.edge_index
        self.valid_graph.node_feature = data.x
        self.valid_graph.node_label = data.y
        self.valid_graph.predict_node = torch.where(data.val_mask)[0]

        self.test_graph = SingleHomogeneousGraphNodeClassification()
        self.test_graph.message_edge = data.edge_index
        self.test_graph.node_feature = data.x
        self.test_graph.node_label = data.y
        self.test_graph.predict_node = torch.where(data.test_mask)[0]
        self.train_graph.count_label()
        self.valid_graph.count_label()
        self.test_graph.count_label()
        print('Done')


class CoraML_CitationFull(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.name = type(self).__name__
        data = CitationFull(root=f'./{self.name}', name='cora_ml')[0]

        self.train_graph = SingleHomogeneousGraphNodeClassification()
        self.train_graph.message_edge = data.edge_index
        self.train_graph.node_feature = data.x
        self.train_graph.node_label = data.y
        self.train_graph.predict_node = torch.where(data.train_mask)[0]

        self.valid_graph = SingleHomogeneousGraphNodeClassification()
        self.valid_graph.message_edge = data.edge_index
        self.valid_graph.node_feature = data.x
        self.valid_graph.node_label = data.y
        self.valid_graph.predict_node = torch.where(data.val_mask)[0]

        self.test_graph = SingleHomogeneousGraphNodeClassification()
        self.test_graph.message_edge = data.edge_index
        self.test_graph.node_feature = data.x
        self.test_graph.node_label = data.y
        self.test_graph.predict_node = torch.where(data.test_mask)[0]
        self.train_graph.count_label()
        self.valid_graph.count_label()
        self.test_graph.count_label()
        print('Done')


class Citeseer_CitationFull(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.name = type(self).__name__
        data = CitationFull(root=f'./{self.name}', name='citeseer')[0]

        self.train_graph = SingleHomogeneousGraphNodeClassification()
        self.train_graph.message_edge = data.edge_index
        self.train_graph.node_feature = data.x
        self.train_graph.node_label = data.y
        self.train_graph.predict_node = torch.where(data.train_mask)[0]

        self.valid_graph = SingleHomogeneousGraphNodeClassification()
        self.valid_graph.message_edge = data.edge_index
        self.valid_graph.node_feature = data.x
        self.valid_graph.node_label = data.y
        self.valid_graph.predict_node = torch.where(data.val_mask)[0]

        self.test_graph = SingleHomogeneousGraphNodeClassification()
        self.test_graph.message_edge = data.edge_index
        self.test_graph.node_feature = data.x
        self.test_graph.node_label = data.y
        self.test_graph.predict_node = torch.where(data.test_mask)[0]
        self.train_graph.count_label()
        self.valid_graph.count_label()
        self.test_graph.count_label()
        print('Done')


class DBLP_CitationFull(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.name = type(self).__name__
        data = CitationFull(root=f'./{self.name}', name='dblp')[0]

        self.train_graph = SingleHomogeneousGraphNodeClassification()
        self.train_graph.message_edge = data.edge_index
        self.train_graph.node_feature = data.x
        self.train_graph.node_label = data.y
        self.train_graph.predict_node = torch.where(data.train_mask)[0]

        self.valid_graph = SingleHomogeneousGraphNodeClassification()
        self.valid_graph.message_edge = data.edge_index
        self.valid_graph.node_feature = data.x
        self.valid_graph.node_label = data.y
        self.valid_graph.predict_node = torch.where(data.val_mask)[0]

        self.test_graph = SingleHomogeneousGraphNodeClassification()
        self.test_graph.message_edge = data.edge_index
        self.test_graph.node_feature = data.x
        self.test_graph.node_label = data.y
        self.test_graph.predict_node = torch.where(data.test_mask)[0]
        self.train_graph.count_label()
        self.valid_graph.count_label()
        self.test_graph.count_label()
        print('Done')


class Pubmed_CitationFull(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.name = type(self).__name__
        data = CitationFull(root=f'./{self.name}', name='pubmed')[0]

        self.train_graph = SingleHomogeneousGraphNodeClassification()
        self.train_graph.message_edge = data.edge_index
        self.train_graph.node_feature = data.x
        self.train_graph.node_label = data.y
        self.train_graph.predict_node = torch.where(data.train_mask)[0]

        self.valid_graph = SingleHomogeneousGraphNodeClassification()
        self.valid_graph.message_edge = data.edge_index
        self.valid_graph.node_feature = data.x
        self.valid_graph.node_label = data.y
        self.valid_graph.predict_node = torch.where(data.val_mask)[0]

        self.test_graph = SingleHomogeneousGraphNodeClassification()
        self.test_graph.message_edge = data.edge_index
        self.test_graph.node_feature = data.x
        self.test_graph.node_label = data.y
        self.test_graph.predict_node = torch.where(data.test_mask)[0]
        self.train_graph.count_label()
        self.valid_graph.count_label()
        self.test_graph.count_label()
        print('Done')


class Computers_Amazon(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.name = type(self).__name__
        data = Amazon(root=f'./{self.name}', name='computers')[0]

        self.train_graph = SingleHomogeneousGraphNodeClassification()
        self.train_graph.message_edge = data.edge_index
        self.train_graph.node_feature = data.x
        self.train_graph.node_label = data.y
        self.train_graph.predict_node = torch.where(data.train_mask)[0]

        self.valid_graph = SingleHomogeneousGraphNodeClassification()
        self.valid_graph.message_edge = data.edge_index
        self.valid_graph.node_feature = data.x
        self.valid_graph.node_label = data.y
        self.valid_graph.predict_node = torch.where(data.val_mask)[0]

        self.test_graph = SingleHomogeneousGraphNodeClassification()
        self.test_graph.message_edge = data.edge_index
        self.test_graph.node_feature = data.x
        self.test_graph.node_label = data.y
        self.test_graph.predict_node = torch.where(data.test_mask)[0]
        self.train_graph.count_label()
        self.valid_graph.count_label()
        self.test_graph.count_label()
        print('Done')


class Photo_Amazon(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.name = type(self).__name__
        data = Amazon(root=f'./{self.name}', name='photo')[0]

        self.train_graph = SingleHomogeneousGraphNodeClassification()
        self.train_graph.message_edge = data.edge_index
        self.train_graph.node_feature = data.x
        self.train_graph.node_label = data.y
        self.train_graph.predict_node = torch.where(data.train_mask)[0]

        self.valid_graph = SingleHomogeneousGraphNodeClassification()
        self.valid_graph.message_edge = data.edge_index
        self.valid_graph.node_feature = data.x
        self.valid_graph.node_label = data.y
        self.valid_graph.predict_node = torch.where(data.val_mask)[0]

        self.test_graph = SingleHomogeneousGraphNodeClassification()
        self.test_graph.message_edge = data.edge_index
        self.test_graph.node_feature = data.x
        self.test_graph.node_label = data.y
        self.test_graph.predict_node = torch.where(data.test_mask)[0]
        self.train_graph.count_label()
        self.valid_graph.count_label()
        self.test_graph.count_label()
        print('Done')


class CS_Coauthor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.name = type(self).__name__
        data = Coauthor(root=f'./{self.name}', name='cs')[0]

        self.train_graph = SingleHomogeneousGraphNodeClassification()
        self.train_graph.message_edge = data.edge_index
        self.train_graph.node_feature = data.x
        self.train_graph.node_label = data.y
        self.train_graph.predict_node = torch.where(data.train_mask)[0]

        self.valid_graph = SingleHomogeneousGraphNodeClassification()
        self.valid_graph.message_edge = data.edge_index
        self.valid_graph.node_feature = data.x
        self.valid_graph.node_label = data.y
        self.valid_graph.predict_node = torch.where(data.val_mask)[0]

        self.test_graph = SingleHomogeneousGraphNodeClassification()
        self.test_graph.message_edge = data.edge_index
        self.test_graph.node_feature = data.x
        self.test_graph.node_label = data.y
        self.test_graph.predict_node = torch.where(data.test_mask)[0]
        self.train_graph.count_label()
        self.valid_graph.count_label()
        self.test_graph.count_label()
        print('Done')


class Physics_Coauthor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.name = type(self).__name__
        data = Coauthor(root=f'./{self.name}', name='physics')[0]

        self.train_graph = SingleHomogeneousGraphNodeClassification()
        self.train_graph.message_edge = data.edge_index
        self.train_graph.node_feature = data.x
        self.train_graph.node_label = data.y
        self.train_graph.predict_node = torch.where(data.train_mask)[0]

        self.valid_graph = SingleHomogeneousGraphNodeClassification()
        self.valid_graph.message_edge = data.edge_index
        self.valid_graph.node_feature = data.x
        self.valid_graph.node_label = data.y
        self.valid_graph.predict_node = torch.where(data.val_mask)[0]

        self.test_graph = SingleHomogeneousGraphNodeClassification()
        self.test_graph.message_edge = data.edge_index
        self.test_graph.node_feature = data.x
        self.test_graph.node_label = data.y
        self.test_graph.predict_node = torch.where(data.test_mask)[0]
        self.train_graph.count_label()
        self.valid_graph.count_label()
        self.test_graph.count_label()
        print('Done')


class WikiCS(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.name = type(self).__name__
        data = torch_geometric.datasets.WikiCS(root=f'./{self.name}')[0]

        self.train_graph = SingleHomogeneousGraphNodeClassification()
        self.train_graph.message_edge = data.edge_index
        self.train_graph.node_feature = data.x
        self.train_graph.node_label = data.y
        self.train_graph.predict_node = torch.where(data.train_mask)[0]

        self.valid_graph = SingleHomogeneousGraphNodeClassification()
        self.valid_graph.message_edge = data.edge_index
        self.valid_graph.node_feature = data.x
        self.valid_graph.node_label = data.y
        self.valid_graph.predict_node = torch.where(data.val_mask)[0]

        self.test_graph = SingleHomogeneousGraphNodeClassification()
        self.test_graph.message_edge = data.edge_index
        self.test_graph.node_feature = data.x
        self.test_graph.node_label = data.y
        self.test_graph.predict_node = torch.where(data.test_mask)[0]
        self.train_graph.count_label()
        self.valid_graph.count_label()
        self.test_graph.count_label()
        print('Done')
