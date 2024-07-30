import torch
from torcheval.metrics import AUC
from callbacks import CSVLogger, ModelCheckpoint, TensorBoard


class Coach:

    def __init__(self, name:str, logs_dir:str, save_dir:str, logger:bool=True, checkpoint:bool=True, tboard:bool=True, device='cuda') -> None:
        self.logs_dir = logs_dir
        self.save_dir = save_dir
        self.device = device
        if logger:
            self.logger = CSVLogger(f'{name}/logs')
        
        if checkpoint:
            self.checkpoint = ModelCheckpoint(f'{name}/models/best.pth')
        
        if tboard:
            self.tensorboard = TensorBoard(f'{name}/tb')

    def fit(self, model, epoches, train_data, val_data):
        pass

    def fit_models(self, model_list,epoches, train_data, val_data):
        pass


class ObjCoach:

    def __init__(self) -> None:
        pass