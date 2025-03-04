from typing import Union
import torch
import torch.nn as nn
from torch_geometric.utils import degree
# from __future__ import annotations
from typing import Tuple, Dict, List
import networkx as nx
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.utils.convert import to_networkx
import matplotlib.pyplot as plt
import numpy as np
import scipy


class Linear(torch.nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.out_dim = out_dim
        self.linear = torch_geometric.nn.Linear(in_channels=-1,
                                                out_channels=self.out_dim,
                                                weight_initializer='kaiming_uniform',
                                                bias=True,
                                                bias_initializer=None)
        self.reset_parameters()

    def reset_parameters(self):
        self.linear.reset_parameters()

    def forward(self, x):
        return self.linear(x)


class Graphormer(nn.Module):
    def __init__(self,
                 num_layer: int,
                 in_node_dim: int,
                 hidden_node_dim: int,
                 in_edge_dim: int,
                 hidden_edge_dim: int,
                 out_dim: int,
                 num_head: int,
                 ffn_dim: int,
                 max_degree: int,
                 max_distance: int,
                 num_graph=1):
        """
        :param num_layer: number of Graphormer layers
        :param in_node_dim: input dimension of node features
        :param hidden_node_dim: hidden dimensions of node features
        :param in_edge_dim: input dimension of edge features
        :param hidden_edge_dim: hidden dimensions of edge features
        :param out_dim: number of output node features
        :param num_head: number of attention heads
        :param max_degree: max in degree of nodes
        :param max_out_degree: max in degree of nodes
        :param max_distance: max pairwise distance between two nodes
        """
        super().__init__()

        self.num_layer = num_layer
        self.in_node_dim = in_node_dim
        self.hidden_node_dim = hidden_node_dim
        self.in_edge_dim = in_edge_dim
        self.hidden_edge_dim = hidden_edge_dim
        self.out_dim = out_dim
        self.num_head = num_head
        self.ffn_dim = ffn_dim
        self.max_degree = max_degree
        self.max_distance = max_distance

        self.node_in_linear = Linear(self.hidden_node_dim)
        self.edge_in_linear = Linear(self.hidden_edge_dim)

        self.centrality_encoding = CentralityEncoding(max_degree=self.max_degree,
                                                      hidden_node_dim=self.hidden_node_dim)

        self.spatial_encoding = SpatialEncoding(max_distance=max_distance)

        self.layers = nn.ModuleList()
        for _ in range(self.num_layer):
            GEL = GraphormerEncoderLayer(node_dim=self.hidden_node_dim,
                                         edge_dim=self.hidden_edge_dim,
                                         n_heads=self.num_head,
                                         ffn_dim=self.ffn_dim,
                                         max_path_distance=self.max_distance)
            self.layers.append(GEL)

        self.node_out_lin = Linear(self.out_dim)

        self.node_paths = {}
        self.node_paths['train'] = {graph_index: None for graph_index in range(num_graph)}
        self.node_paths['valid'] = {graph_index: None for graph_index in range(num_graph)}
        self.node_paths['test'] = {graph_index: None for graph_index in range(num_graph)}

        self.edge_paths = {}
        self.edge_paths['train'] = {graph_index: None for graph_index in range(num_graph)}
        self.edge_paths['valid'] = {graph_index: None for graph_index in range(num_graph)}
        self.edge_paths['test'] = {graph_index: None for graph_index in range(num_graph)}

    def forward(self, x, edge_index, edge_attribution=None, phase='train', graph_index=0) -> torch.Tensor:
        node_emb = x.clone()
        if edge_attribution is None:
            edge_emb = edge_attribution
        else:
            edge_emb = edge_attribution.clone()

        assert phase in ['train', 'valid', 'test']
        if self.node_paths[phase][graph_index] is None and self.edge_paths[phase][graph_index] is None:
            node_paths, edge_paths = self.shortest_path_distance(edge_index=edge_index)
            self.node_paths[phase][graph_index] = node_paths
            self.edge_paths[phase][graph_index] = edge_paths
        else:
            node_paths = self.node_paths[phase][graph_index]
            edge_paths = self.edge_paths[phase][graph_index]

        if node_emb is not None: node_emb = self.node_in_linear(node_emb)
        if edge_emb is not None: edge_emb = self.edge_in_linear(edge_emb)

        node_emb = self.centrality_encoding(node_emb, edge_index)
        b = self.spatial_encoding(node_emb, edge_index)

        for layer in self.layers:
            node_emb = layer(node_emb, edge_emb, b, edge_paths)

        node_emb = self.node_out_lin(node_emb)

        return node_emb

    def shortest_path_distance(self, edge_index) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
        edges = edge_index.detach().cpu().T.numpy().tolist()
        G = nx.DiGraph()
        G.add_edges_from(edges)
        node_paths, edge_paths = self.all_pairs_shortest_path(G)
        return node_paths, edge_paths

    def batched_shortest_path_distance(self, data) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
        graphs = [to_networkx(sub_data) for sub_data in data.to_data_list()]
        relabeled_graphs = []
        shift = 0
        for i in range(len(graphs)):
            num_nodes = graphs[i].number_of_nodes()
            relabeled_graphs.append(nx.relabel_nodes(graphs[i], {i: i + shift for i in range(num_nodes)}))
            shift += num_nodes

        paths = [self.all_pairs_shortest_path(G) for G in relabeled_graphs]
        node_paths = {}
        edge_paths = {}

        for path in paths:
            for k, v in path[0].items():
                node_paths[k] = v
            for k, v in path[1].items():
                edge_paths[k] = v

        return node_paths, edge_paths

    def all_pairs_shortest_path(self, G) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
        paths = {}
        for n in G:
            key = n
            value = self.floyd_warshall_source_to_all(G, n)
            paths[key] = value
        node_paths = {n: paths[n][0] for n in paths}
        edge_paths = {n: paths[n][1] for n in paths}
        return node_paths, edge_paths

    def floyd_warshall_source_to_all(self, G, source, cutoff=None):
        if source not in G:
            raise nx.NodeNotFound("Source {} not in G".format(source))

        edges = {edge: i for i, edge in enumerate(G.edges())}

        level = 0  # the current level
        nextlevel = {source: 1}  # list of nodes to check at next level
        node_paths = {source: [source]}  # paths dictionary  (paths to key from source)
        edge_paths = {source: []}

        while nextlevel:
            thislevel = nextlevel
            nextlevel = {}
            for v in thislevel:
                for w in G[v]:
                    if w not in node_paths:
                        node_paths[w] = node_paths[v] + [w]
                        edge_paths[w] = edge_paths[v] + [edges[tuple(node_paths[w][-2:])]]
                        nextlevel[w] = 1

            level = level + 1

            if (cutoff is not None and cutoff <= level):
                break

        return node_paths, edge_paths


class CentralityEncoding(torch.nn.Module):
    def __init__(self, max_degree, hidden_node_dim: int):
        super().__init__()
        self.hidden_node_dim = hidden_node_dim
        self.max_degree = max_degree

        # z_in
        self.in_degree_emb = torch.nn.Parameter(torch.randn((max_degree, self.hidden_node_dim)))

        # z_out
        self.out_degree_emb = torch.nn.Parameter(torch.randn((max_degree, self.hidden_node_dim)))

    def forward(self, node_emb: torch.Tensor, edge_index: torch.LongTensor) -> torch.Tensor:
        num_nodes = node_emb.shape[0]

        in_degree = torch_geometric.utils.degree(index=edge_index[1], num_nodes=num_nodes).long()
        in_degree = self.decrease_to_max_value(degree=in_degree, max_degree=self.max_degree)

        out_degree = torch_geometric.utils.degree(index=edge_index[0], num_nodes=num_nodes).long()
        out_degree = self.decrease_to_max_value(degree=out_degree, max_degree=self.max_degree)

        in_degree = in_degree - 1
        out_degree = out_degree - 1

        node_emb = node_emb + self.in_degree_emb[in_degree] + self.out_degree_emb[out_degree]

        return node_emb

    def decrease_to_max_value(self, degree, max_degree):
        degree[degree > max_degree] = max_degree

        return degree


class SpatialEncoding(nn.Module):
    def __init__(self, max_distance: int):
        super().__init__()
        self.max_distance = max_distance

        # b
        self.distance_emb = nn.Parameter(torch.randn(self.max_distance))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        num_source = x.shape[0]
        num_target = x.shape[0]
        sparse_size = (num_source, num_target)

        edge_weight = np.ones(edge_index.shape[1])
        source_index = edge_index[0].detach().cpu().numpy()
        target_index = edge_index[1].detach().cpu().numpy()
        arg1 = (edge_weight, (source_index, target_index))

        adj = scipy.sparse.csr_matrix(arg1=arg1, shape=sparse_size)

        distance = scipy.sparse.csgraph.shortest_path(adj, directed=True)
        distance = distance.astype(np.int64)
        distance = torch.tensor(distance, device=edge_index.device, dtype=edge_index.dtype)
        mask = distance == np.inf
        distance[mask] = -1

        distance_emb = torch.zeros((x.shape[0], x.shape[0])).to(self.distance_emb.device)
        distance_emb = self.distance_emb[distance]

        return distance_emb


class GraphormerEncoderLayer(nn.Module):
    def __init__(self, node_dim, edge_dim, n_heads, ffn_dim, max_path_distance):
        """
        :param node_dim: node feature matrix input number of dimension
        :param edge_dim: edge feature matrix input number of dimension
        :param n_heads: number of attention heads
        """
        super().__init__()

        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.n_heads = n_heads
        self.ffn_dim = ffn_dim

        self.attention = GraphormerMultiHeadAttention(
            dim_in=node_dim,
            dim_k=node_dim,
            dim_q=node_dim,
            num_heads=n_heads,
            edge_dim=edge_dim,
            max_path_distance=max_path_distance,
        )
        self.ln_1 = nn.LayerNorm(self.node_dim)
        self.ln_2 = nn.LayerNorm(self.node_dim)
        self.ffn = nn.Sequential(
            nn.Linear(self.node_dim, self.ffn_dim),
            nn.GELU(),
            nn.Linear(self.ffn_dim, self.node_dim)
        )

    def forward(self,
                x: torch.Tensor,
                edge_attr: torch.Tensor,
                b: torch,
                edge_paths,
                ptr) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        h′(l) = MHA(LN(h(l−1))) + h(l−1)
        h(l) = FFN(LN(h′(l))) + h′(l)

        :param x: node feature matrix
        :param edge_attr: edge feature matrix
        :param b: spatial Encoding matrix
        :param edge_paths: pairwise node paths in edge indexes
        :param ptr: batch pointer that shows graph indexes in batch of graphs
        :return: torch.Tensor, node embeddings after Graphormer layer operations
        """
        x_prime = self.attention(self.ln_1(x), edge_attr, b, edge_paths, ptr) + x
        x_new = self.ffn(self.ln_2(x_prime)) + x_prime

        return x_new


# FIX: PyG attention instead of regular attention, due to specificity of GNNs
class GraphormerMultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, dim_in: int, dim_q: int, dim_k: int, edge_dim: int, max_path_distance: int):
        """
        :param num_heads: number of attention heads
        :param dim_in: node feature matrix input number of dimension
        :param dim_q: query node feature matrix input number dimension
        :param dim_k: key node feature matrix input number of dimension
        :param edge_dim: edge feature matrix number of dimension
        """
        super().__init__()
        self.heads = nn.ModuleList(
            [GraphormerAttentionHead(dim_in, dim_q, dim_k, edge_dim, max_path_distance) for _ in range(num_heads)]
        )
        self.linear = nn.Linear(num_heads * dim_k, dim_in)

    def forward(self,
                x: torch.Tensor,
                edge_attr: torch.Tensor,
                b: torch.Tensor,
                edge_paths,
                ptr) -> torch.Tensor:
        """
        :param x: node feature matrix
        :param edge_attr: edge feature matrix
        :param b: spatial Encoding matrix
        :param edge_paths: pairwise node paths in edge indexes
        :param ptr: batch pointer that shows graph indexes in batch of graphs
        :return: torch.Tensor, node embeddings after all attention heads
        """
        return self.linear(
            torch.cat([
                attention_head(x, edge_attr, b, edge_paths, ptr) for attention_head in self.heads
            ], dim=-1)
        )


class GraphormerAttentionHead(nn.Module):
    def __init__(self, dim_in: int, dim_q: int, dim_k: int, edge_dim: int, max_path_distance: int):
        """
        :param dim_in: node feature matrix input number of dimension
        :param dim_q: query node feature matrix input number dimension
        :param dim_k: key node feature matrix input number of dimension
        :param edge_dim: edge feature matrix number of dimension
        """
        super().__init__()
        self.edge_encoding = EdgeEncoding(edge_dim, max_path_distance)

        self.q = nn.Linear(dim_in, dim_q)
        self.k = nn.Linear(dim_in, dim_k)
        self.v = nn.Linear(dim_in, dim_k)

    def forward(self,
                x: torch.Tensor,
                edge_attr: torch.Tensor,
                b: torch.Tensor,
                edge_paths,
                ptr=None) -> torch.Tensor:
        """
        :param query: node feature matrix
        :param key: node feature matrix
        :param value: node feature matrix
        :param edge_attr: edge feature matrix
        :param b: spatial Encoding matrix
        :param edge_paths: pairwise node paths in edge indexes
        :param ptr: batch pointer that shows graph indexes in batch of graphs
        :return: torch.Tensor, node embeddings after attention operation
        """
        batch_mask_neg_inf = torch.full(size=(x.shape[0], x.shape[0]), fill_value=-1e6).to(
            next(self.parameters()).device)
        batch_mask_zeros = torch.zeros(size=(x.shape[0], x.shape[0])).to(next(self.parameters()).device)

        # OPTIMIZE: get rid of slices: rewrite to torch
        if type(ptr) == type(None):
            batch_mask_neg_inf = torch.ones(size=(x.shape[0], x.shape[0])).to(next(self.parameters()).device)
            batch_mask_zeros += 1
        else:
            for i in range(len(ptr) - 1):
                batch_mask_neg_inf[ptr[i]:ptr[i + 1], ptr[i]:ptr[i + 1]] = 1
                batch_mask_zeros[ptr[i]:ptr[i + 1], ptr[i]:ptr[i + 1]] = 1

        query = self.q(x)
        key = self.k(x)
        value = self.v(x)

        c = self.edge_encoding(x, edge_attr, edge_paths)
        a = self.compute_a(key, query, ptr)
        a = (a + b + c) * batch_mask_neg_inf
        softmax = torch.softmax(a, dim=-1) * batch_mask_zeros
        x = softmax.mm(value)
        return x

    def compute_a(self, key, query, ptr=None):
        if type(ptr) == type(None):
            a = query.mm(key.transpose(0, 1)) / query.size(-1) ** 0.5
        else:
            a = torch.zeros((query.shape[0], query.shape[0]), device=key.device)
            for i in range(len(ptr) - 1):
                a[ptr[i]:ptr[i + 1], ptr[i]:ptr[i + 1]] = query[ptr[i]:ptr[i + 1]].mm(
                    key[ptr[i]:ptr[i + 1]].transpose(0, 1)) / query.size(-1) ** 0.5

        return a


class EdgeEncoding(nn.Module):
    def __init__(self, edge_dim: int, max_path_distance: int):
        """
        :param edge_dim: edge feature matrix number of dimension
        """
        super().__init__()
        self.edge_dim = edge_dim
        self.max_path_distance = max_path_distance
        self.edge_vector = nn.Parameter(torch.randn(self.max_path_distance, self.edge_dim))

    def forward(self, x: torch.Tensor, edge_attr: torch.Tensor, edge_paths) -> torch.Tensor:
        """
        :param x: node feature matrix
        :param edge_attr: edge feature matrix
        :param edge_paths: pairwise node paths in edge indexes
        :return: torch.Tensor, Edge Encoding matrix
        """
        cij = torch.zeros((x.shape[0], x.shape[0])).to(next(self.parameters()).device)

        for src in edge_paths:
            for dst in edge_paths[src]:
                path_ij = edge_paths[src][dst][:self.max_path_distance]
                weight_inds = [i for i in range(len(path_ij))]
                cij[src][dst] = self.dot_product(self.edge_vector[weight_inds], edge_attr[path_ij]).mean()

        cij = torch.nan_to_num(cij)
        return cij

    def dot_product(self, x1, x2) -> torch.Tensor:
        return (x1 * x2).sum(dim=1)


node_feature = torch.randn(8, 32)
edge_index = torch.tensor([
    [2, 3, 3, 4, 5, 6, 7, 7, 1],
    [0, 0, 1, 1, 2, 3, 4, 5, 2]],
)
edge_index = torch_geometric.utils.to_undirected(edge_index)

# G = nx.DiGraph()
# G.add_edges_from(edge_index.T.numpy().tolist())
# pos = nx.spring_layout(G)
# nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=10, font_color='black',
#         font_weight='bold', edge_color='gray')
# plt.savefig('toygraph.png')
# plt.close()

model = Graphormer(num_layer=1,
                   in_node_dim=-1,
                   hidden_node_dim=32,
                   in_edge_dim=-1,
                   hidden_edge_dim=32,
                   out_dim=7,
                   num_head=1,
                   ffn_dim=32,
                   max_degree=node_feature.shape[0],
                   max_distance=node_feature.shape[0],
                   num_graph=1)

zzz = model.forward(x=node_feature,
                    edge_index=edge_index,
                    phase='train',
                    graph_index=0)
