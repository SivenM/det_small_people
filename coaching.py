import os
from tqdm.auto import tqdm
from typing import Tuple
import torch
from torch import nn
#from torcheval.metrics.functional import binary_accuracy, multiclass_accuracy, multiclass_recall, multiclass_precision
from metrics import MultiMetrics
from callbacks import CSVLogger, ModelCheckpoint, TensorBoard
import utils
from loguru import logger

BASE_DIR = "runs"


class Coach:

    def __init__(
            self, name:str, 
            save_dir:str=None, 
            metric:str='bin_acc', 
            logger:bool=True, 
            checkpoint:bool=True, 
            tboard:bool=True, 
            device='cuda', 
            debug:bool=False
            ) -> None:
        if save_dir:
            utils.mkdir(save_dir)
            utils.mkdir(f'{save_dir}/{name}')
            self.logs_dir = f'{save_dir}/{name}/logs'
            self.save_dir = f'{save_dir}/{name}/models/'
        else:
            #utils.mkdir(f'{BASE_DIR}/{name}')
            self.logs_dir = None#f'{BASE_DIR}/{name}/logs'
            self.save_dir = None#f'{BASE_DIR}/{name}/models/'
        self.device = device
        
        if metric == 'multi_acc':
            self.metric = MultiMetrics()
        elif metric == 'bin_acc':               #TODO
            pass
        else:
            raise 'Wrong metric'
            
        if logger:
            self.logger = CSVLogger(self.logs_dir)
        else:
            self.logger = None

        if checkpoint:
            self.checkpoint = ModelCheckpoint(self.save_dir)
        else:
            self.checkpoint = None

        if tboard:
            self.tboard = TensorBoard(f'{save_dir}/{name}/tb', 
                                      ['loss',
                                       'v_loss,',
                                       't_acc',
                                       't_precision',
                                       't_recall',
                                       'v_acc',
                                       'v_precision',
                                       'v_recall',
                                       ]
                                    )
        else:
            self.tboard = None
        
        self.history = {
            't_loss': [],
            'v_loss': [],
            't_acc': [],
            'v_acc': []
        }
        self.device = device
        self.debug = debug

    def __create_dirs(self, name):
        exp_dir = os.path.join(BASE_DIR, name)
        logs_dir = os.path.join(exp_dir, 'logs')
        models_dir = os.path.join(exp_dir, 'models')
        for path in [exp_dir, logs_dir, ]:
            utils.mkdir(os.path.join(BASE_DIR, name))

    def _to_cuda(self, targets:list) -> list:
        for target in targets:
            target['bbox'] = target['bbox'].to('cuda')
            target['labels'] = target['labels'].to('cuda')
        return targets
    
    def train_step(self, epoch, data) -> Tuple[float, float]:
        self.model.train()
        self.loss_fn.train()
        t_loss = 0.
        t_acc = 0.
        t_prec = 0.
        t_recall = 0.

        progress_bar = tqdm(
            enumerate(data),
            desc=f"train epoch {epoch}",
            total=(len(data)),
            disable=False
        )
        #print(f'train step!')
        for batch, (sample, label) in progress_bar:
            self.optimizer.zero_grad()
            sample, label = sample.to(self.device), label.to(self.device)
            pred = self.model(sample)
            pred = pred
            #print(f'{pred=}\n{label}\n\n')
            loss = self.loss_fn(pred, label)
            t_loss += loss.item()
            #if self.debug:
            #print(f'loss: {t_loss}')
            metrics = self.metric(pred, label)
            t_acc += metrics['acc']
            t_prec += metrics['precision']
            t_recall += metrics['recall']
            loss.backward()
            self.optimizer.step()
            progress_bar.set_postfix(
                {
                    'train_loss': t_loss / (batch + 1),
                    'train_acc': t_acc / (batch + 1)
                }
            )
        t_loss = t_loss / len(data)
        t_acc = t_acc / len(data)
        t_prec = t_prec / len(data)
        t_recall = t_recall / len(data)
        return t_loss, {'acc': t_acc, 'precision': t_prec, 'recall':t_recall}

    def val_step(self, epoch, data) -> Tuple[float, float]:
        self.model.eval()
        self.loss_fn.eval()
        v_loss, v_acc = 0, 0
        v_prec = 0.
        v_recall = 0.
        progress_bar = tqdm(
            enumerate(data),
            desc=f"val epoch {epoch}",
            total=(len(data)),
            disable=False
        )

        with torch.no_grad():
            for batch, (sample, label) in progress_bar:
                sample, label= sample.to(self.device), label.to('cuda').squeeze(-1)
                pred = self.model(sample)
                pred = pred.squeeze(-1)
                #print(f'{pred=}\n{label}\n\n')
                loss = self.loss_fn(pred, label)
                v_loss += loss.item()
                metrics = self.metric(pred, label)
                v_acc += metrics['acc']
                v_prec += metrics['precision']
                v_recall += metrics['recall']
                progress_bar.set_postfix(
                    {
                        'val_loss': v_loss / (batch + 1),
                        'val_acc': v_acc / (batch + 1)
                    }
                )
        v_loss = v_loss / len(data)
        v_acc = v_acc / len(data)
        v_prec = v_prec / len(data)
        v_recall = v_recall / len(data)
        return v_loss, {'acc': v_acc, 'precision': v_prec, 'recall':v_recall}

    def clear_history(self):
        self.history['t_loss'] = []
        self.history['v_loss'] = []
        self.history['t_metrics'] = []
        self.history['v_metrics'] = []

    def update_history(self, t_loss, v_loss, t_metrics, v_metrics):
        self.history['t_loss'].append(t_loss)
        self.history['v_loss'].append(v_loss)
        self.history['t_metrics'].append(t_metrics)
        self.history['v_metrics'].append(v_metrics)

    def update_callbacks(self, epoch, t_loss, v_loss, t_metrics, v_metrics):
        print('callbacks started')
        if self.logger:
            print('csv logger run')
            self.logger.log(epoch, t_loss, v_loss, t_metrics, v_metrics)
        
        if self.checkpoint:
            self.checkpoint.save(self.model, epoch, v_loss)

        if self.tboard:
            self.tboard.add([t_loss, v_loss, 
                             t_metrics['acc'], t_metrics['precision'], t_metrics['recall'], 
                             v_metrics['acc'], v_metrics['precision'], v_metrics['recall'],], 
                             epoch
                             )

    def plot_history(self):
        pass

    def fit(self, epoches, model, loss_fn, lr, train_data, val_data=None, start_epoch=0):
        self.clear_history()
        self.model = model.to(self.device)
        self.loss_fn = loss_fn
        self.loss_fn.debug = self.debug
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        print('start fit!')
        for epoch in tqdm(range(start_epoch, epoches)):
            print(f'{epoch=}')
            t_loss, t_metrics = self.train_step(epoch, train_data)
            if val_data:
                v_loss, v_metrics = self.val_step(epoch, val_data)
            else:
                v_loss, v_metrics = 0., 0., 0., 0.
            
            self.update_callbacks(epoch, t_loss, v_loss, t_metrics, v_metrics)
            self.update_history(t_loss, v_loss, t_metrics, v_metrics)
        if self.save_dir:
            torch.save(self.model.state_dict(), os.path.join(self.save_dir, f'last_acc{v_metrics["acc"]:.2f}.pth'))
        return self.history


class DetrLocCoach(Coach):
    def __init__(self, name: str, save_dir: str = None, metric: str = 'multi_acc', logger: bool = True, checkpoint: bool = True, tboard: bool = False, device='cuda', debug: bool = False) -> None:
        super().__init__(name, save_dir, metric, logger, checkpoint, tboard, device, debug)

    def _to_cuda(self, targets:list) -> list:
        out = []
        for target in targets:
            out.append(target.to('cuda'))
        return out

    def train_step(self, epoch, data) -> Tuple[float, float]:
        self.model.train()
        self.loss_fn.train()
        t_loss, t_acc = 0., 0.

        progress_bar = tqdm(
            enumerate(data),
            desc=f"train epoch {epoch}",
            total=(len(data)),
            disable=False
        )

        for batch, (sample, label) in progress_bar:
            #if torch.cat([v for v in label]).shape[0] == 0:
            #    continue
            sample, label = sample.to(self.device), self._to_cuda(label)
            pred = self.model(sample)
            loss = self.loss_fn(pred, label)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            #print(loss.item())
            t_loss += loss.item()
            progress_bar.set_postfix(
                {
                    'train_loss': t_loss / (batch + 1)
                }
            )
        t_loss = t_loss / len(data)
        return t_loss, t_acc
    
    def val_step(self, epoch, data) -> Tuple[float, float]:
        self.model.eval()
        self.loss_fn.eval()
        v_loss, v_acc = 0, 0
        progress_bar = tqdm(
            enumerate(data),
            desc=f"val epoch {epoch}",
            total=(len(data)),
            disable=False
        )

        with torch.no_grad():
            for batch, (sample, label) in progress_bar:
                if torch.cat([v for v in label]).shape[0]  == 0:
                    continue
                sample, label= sample.to(self.device), self._to_cuda(label)
                pred = self.model(sample)
                loss = self.loss_fn(pred, label)
                v_loss += loss.item()
                progress_bar.set_postfix(
                    {
                        'val_loss': v_loss / (batch + 1),
                    }
                )
        v_loss = v_loss / len(data)
        return v_loss, v_acc



class LocCoach(Coach):
    def __init__(self, name: str, save_dir: str = None, metric: str = 'bin_acc', logger: bool = True, checkpoint: bool = True, tboard: bool = True, device='cuda', debug: bool = False) -> None:
        super().__init__(name, save_dir, metric, logger, checkpoint, tboard, device, debug)

    def train_step(self, epoch, data) -> Tuple[float, float]:
        self.model.train()
        self.loss_fn.train()
        t_loss = 0.
        t_acc = 0.

        progress_bar = tqdm(
            enumerate(data),
            desc=f"train epoch {epoch}",
            total=(len(data)),
            disable=False
        )

        for batch, (sample, label) in progress_bar:
            sample, label = sample.to(self.device), label.to(self.device)
            pred = self.model(sample)
            loss = self.loss_fn(pred, label)
            t_loss += loss.item()
            #if self.debug:
            #print(f'loss: {t_loss}')
            #t_acc += self.metric(pred, label).item()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            progress_bar.set_postfix(
                {
                    'train_loss': t_loss / (batch + 1),
                    #'train_acc': t_acc / (batch + 1)
                }
            )
        t_loss = t_loss / len(data)
        #t_acc = t_acc / len(data)
        return t_loss, t_acc

    def val_step(self, epoch, data) -> Tuple[float, float]:
        self.model.eval()
        self.loss_fn.eval()
        v_loss, v_acc = 0, 0
        progress_bar = tqdm(
            enumerate(data),
            desc=f"val epoch {epoch}",
            total=(len(data)),
            disable=False
        )

        with torch.no_grad():
            for batch, (sample, label) in progress_bar:
                sample, label= sample.to(self.device), label.to('cuda')
                pred = self.model(sample)
                pred = pred
                loss = self.loss_fn(pred, label)
                v_loss += loss.item()
                #v_acc += self.metric(pred, label.squeeze(-1)).item()
                progress_bar.set_postfix(
                    {
                        'val_loss': v_loss / (batch + 1),
                        #'val_acc': v_acc / (batch + 1)
                    }
                )
        v_loss = v_loss / len(data)
        #v_acc = v_acc / len(data)
        return v_loss, v_acc


class SeqDetCoach(Coach):
    def __init__(
            self, name: str, save_dir: str = None, metric: str = 'multi_acc', 
            logger: bool = True, checkpoint: bool = True, tboard: bool = False,
              device='cuda', debug: bool = False, progress_bar=True) -> None:
        super().__init__(name, save_dir, metric, logger, checkpoint, tboard, device, debug)
        if progress_bar:
            self.disable = False
        else:
            self.disable = True
    def _to_cuda(self, targets:list) -> list:
        out = []
        for target in targets:
            target['bbox'] = target['bbox'].to('cuda')
            target['labels'] = target['labels'].to('cuda')
            out.append(target)
        return out

    def train_step(self, epoch, data) -> Tuple[float, float]:
        self.model.train()
        self.loss_fn.train()
        t_loss, t_acc, t_iou = 0., 0., 0.

        progress_bar = tqdm(
            enumerate(data),
            desc=f"train epoch {epoch}",
            total=(len(data)),
            disable=self.disable
        )

        for batch, (sample, label) in progress_bar:
            #if torch.cat([v for v in label]).shape[0] == 0:
            #    continue
            sample, label = sample.to(self.device), self._to_cuda(label)
            pred = self.model(sample)
            loss, iou = self.loss_fn(pred, label)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            #print(loss.item())
            t_loss += loss.item()
            t_iou += iou.mean().item()
            progress_bar.set_postfix(
                {
                    'train_loss': t_loss / (batch + 1),
                    'train_iou': t_iou / (batch + 1)
                }
            )
        t_loss = t_loss / len(data)
        t_iou = t_iou / len(data)
        return t_loss, t_acc, t_iou
    
    def val_step(self, epoch, data) -> Tuple[float, float]:
        self.model.eval()
        self.loss_fn.eval()
        v_loss, v_acc, v_iou = 0, 0, 0
        progress_bar = tqdm(
            enumerate(data),
            desc=f"val epoch {epoch}",
            total=(len(data)),
            disable=self.disable
        )

        with torch.no_grad():
            for batch, (sample, label) in progress_bar:
                #if torch.cat([v for v in label]).shape[0]  == 0:
                #    continue
                sample, label= sample.to(self.device), self._to_cuda(label)
                pred = self.model(sample)
                loss, iou = self.loss_fn(pred, label)
                v_loss += loss.item()
                v_iou += iou.mean().item()
                progress_bar.set_postfix(
                    {
                        'val_loss': v_loss / (batch + 1),
                        'val_iou': v_iou / (batch + 1),
                    }
                )
        v_loss = v_loss / len(data)
        v_iou = v_iou / len(data)
        return v_loss, v_acc, v_iou