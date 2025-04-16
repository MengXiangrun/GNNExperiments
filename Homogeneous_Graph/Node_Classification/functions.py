import pandas as pd
import torch
import torch_geometric
from torch.nn import Module, ModuleList, ModuleDict
from torch.nn import Parameter, ParameterList, ParameterDict
from torch.nn import BatchNorm1d, BatchNorm2d, BatchNorm3d
from torch.nn import LayerNorm
from torch.nn import Dropout, Dropout1d, Dropout2d, Dropout3d
from torch.nn import BCELoss, MSELoss, CrossEntropyLoss, BCEWithLogitsLoss
# torch_geometric.transforms.add_positional_encoding
from sklearn.metrics import accuracy_score, f1_score, classification_report, precision_score, recall_score
from datetime import datetime


def round_float(input_dict):
    rounded_dict = {}
    for key, value in input_dict.items():
        if isinstance(value, float):
            rounded_dict[key] = round(value, 4)
        else:
            rounded_dict[key] = value
    return rounded_dict


def node_classification_metrics(true, pred):
    num_node, num_label = true.shape
    assert pred.shape[0] == num_node
    assert pred.shape[1] == num_label

    true = true.max(1)[1]
    pred = pred.max(1)[1]

    true = true.detach().cpu().numpy()
    pred = pred.detach().cpu().numpy()

    metrics = {}
    metrics['accuracy'] = accuracy_score(y_true=true, y_pred=pred)
    metrics['macro_precision'] = precision_score(y_true=true, y_pred=pred, average='macro', zero_division=0)
    metrics['micro_precision'] = precision_score(y_true=true, y_pred=pred, average='micro', zero_division=0)
    metrics['macro_recall'] = recall_score(y_true=true, y_pred=pred, average='macro', zero_division=0)
    metrics['micro_recall'] = recall_score(y_true=true, y_pred=pred, average='micro', zero_division=0)
    metrics['macro_f1'] = f1_score(y_true=true, y_pred=pred, average='macro', zero_division=0)
    metrics['micro_f1'] = f1_score(y_true=true, y_pred=pred, average='micro', zero_division=0)

    metrics = round_float(input_dict=metrics)

    return metrics


def train(model, graph, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    device = next(model.parameters()).device

    node_embed = model.encoder.forward(node_embed=graph.node_feature.to(device=device),
                                       message_edge=graph.message_edge.to(device=device))
    pred_label = model.preditor.forward(node_embed)

    # prediction
    true_label = torch.nn.functional.one_hot(input=graph.node_label, num_classes=graph.num_node_label)
    true_label = true_label.to(dtype=pred_label.dtype, device=device)
    true_label = true_label[graph.predict_node.to(device=device)]
    pred_label = pred_label[graph.predict_node.to(device=device)]

    # loss
    if isinstance(criterion, (CrossEntropyLoss, BCELoss, BCEWithLogitsLoss)):
        loss = criterion(input=pred_label, target=true_label)

    # metrics
    metrics = node_classification_metrics(pred=pred_label, true=true_label)

    print('train')
    print(f'loss {loss.item():.4f}')
    print(f'{metrics}')

    loss.backward()
    optimizer.step()

    return model, optimizer, criterion


def valid(model, graph, stop):
    with torch.no_grad():
        model.eval()
        device = next(model.parameters()).device

        node_embed = model.encoder.forward(node_embed=graph.node_feature.to(device=device),
                                           message_edge=graph.message_edge.to(device=device))
        pred_label = model.preditor.forward(node_embed)

        # prediction
        true_label = torch.nn.functional.one_hot(input=graph.node_label, num_classes=graph.num_node_label)
        true_label = true_label.to(dtype=pred_label.dtype, device=device)
        true_label = true_label[graph.predict_node.to(device=device)]
        pred_label = pred_label[graph.predict_node.to(device=device)]

        # metrics
        metrics = node_classification_metrics(pred=pred_label, true=true_label)
        print('valid')
        print(f'{metrics}')

        stop.record(model=model, now_val_loss=-metrics['accuracy'])

    return model, stop


def test(model, graph):
    with torch.no_grad():
        model.eval()
        device = next(model.parameters()).device

        node_embed = model.encoder.forward(node_embed=graph.node_feature.to(device=device),
                                           message_edge=graph.message_edge.to(device=device))
        pred_label = model.preditor.forward(node_embed)

        # prediction
        true_label = torch.nn.functional.one_hot(input=graph.node_label, num_classes=graph.num_node_label)
        true_label = true_label.to(dtype=pred_label.dtype, device=device)
        true_label = true_label[graph.predict_node.to(device=device)]
        pred_label = pred_label[graph.predict_node.to(device=device)]

        # metrics
        metrics = node_classification_metrics(pred=pred_label, true=true_label)

        print('test')
        print(f'{metrics}')

    return model, metrics
