import random
import numpy as np
import torch

class EarlyStopping():
    def __init__(self, patience, threshold):
        self.val_loss_list = list()
        self.best_model_parameter = 0
        self.total_patience = patience
        self.patience = patience
        self.threshold = threshold
        self.count = 0
        self.is_stop = False
        self.min_val_loss = None
        self.save_path = None

    def record(self, model, now_val_loss):
        if self.min_val_loss is None:
            self.min_val_loss = now_val_loss

        if now_val_loss < self.min_val_loss:
            self.best_model_parameter = model.state_dict()
            self.patience = self.total_patience
            self.min_val_loss = now_val_loss

        if now_val_loss >= self.min_val_loss:
            self.patience = self.patience - 1

        if self.patience <= 0:
            self.is_stop = True

        print('patience:', self.patience)

    def save(self, dataset_name, model_name):
        self.save_path = (dataset_name,model_name)
        self.save_path = f'{self.save_path}.pth'
        torch.save(self.best_model_parameter, self.save_path)
        print(f'best_model_parameter saved: {self.save_path}')

    def load(self, model, path=None):
        if path is None:
            path = self.save_path
        self.best_model_parameter = torch.load(path)
        model.load_state_dict(self.best_model_parameter)

        return model

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False