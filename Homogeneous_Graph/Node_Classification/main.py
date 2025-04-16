import random
import numpy as np
import torch
from sklearn.metrics import accuracy_score
from basic_gnn import *
from dataset import *
import pandas as pd
from tools import *
from functions import *

set_seed(seed=1)
device = torch.device("cuda")

result = {}
mean_test_accuracy = []

data = Cora_Planetoid()
train_graph, valid_graph, test_graph = data.train_graph, data.valid_graph, data.test_graph

config = GNNConfig()
encoder = GCN(config=config)
predictor = Linear(out_dim=test_graph.num_node_label)
model = Model(encoder=encoder, preditor=predictor)
model = model.to(device=device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), weight_decay=0.0005, lr=0.0005)
stop = EarlyStopping(patience=30, threshold=0.0001)

for epoch in range(10000):
    if stop.is_stop: break
    print('epoch:', epoch)
    model, optimizer, criterion = train(model=model, optimizer=optimizer, criterion=criterion, graph=train_graph)
    with torch.no_grad():
        model, stop = valid(model=model, stop=stop, graph=valid_graph)
    print()

stop.save(model_name=model.encoder.name, dataset_name=data.name)

# test
with torch.no_grad():
    model = stop.load(model=model)
    model, metrics = test(model=model, graph=test_graph)

current_time = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
metrics['time'] = current_time
metrics['model'] = model.encoder.name
metrics['task'] = 'node_classification'
metrics['dataset'] = data.name
metrics = pd.DataFrame(metrics, index=[0])
filename = f'{current_time}.xlsx'
metrics.to_excel(filename, index=False)
