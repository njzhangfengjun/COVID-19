#!/usr/bin/env python3
import fire
from tqdm import tqdm
import pandas as pd
import numpy as npfire
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from fastai.metrics import add_metrics
from fastai.callback import Callback

from PIL import Image
from sklearn.metrics import roc_auc_score


def parse_device(device):
    if device is None:
        devices = []
    elif not ',' in device:
        devices = [device]
    else:
        devices = device.split(',')
    return devices


def image_verify(path):
    try:
        if not path.is_file():
            path = None
        else:
            Image.open(path).verify()
    except Exception as e:
        print(path)
        print(e)
        path = None
    return path


class AUCk(Callback):
    def __init__(self, class_y):
        self.name = f'AUC({class_y})'
        self._class_y = class_y


    def on_epoch_begin(self, **kwargs):
        self._label = []
        self._score = []
        

    def on_batch_end(self, last_output, last_target, **kwargs):
        '''
        last_output: (B)
        last_target: (B,k)
        '''
        self._label += [(last_target == self._class_y).to('cpu').data.numpy()]
        self._score += [last_output[:,self._class_y].to('cpu').data.numpy()]    


    def on_epoch_end(self, last_metrics, **kwargs):
        label = np.concatenate(self._label)
        score = np.concatenate(self._score)
        if len(set(label)) != 2:
            auc = 0.5
        else:
            score = np.concatenate(self._score)
            auc = roc_auc_score(label,score)
        return add_metrics(last_metrics, auc)


class VolumeClassification(object):
    def __init__(self, lr=1e-3, n_epoch=10, 
                    input_channels=8, sample_duration=64, sample_size=128,
                    batch_size=32, num_workers=0, device=None,):
        super(VolumeClassification, self).__init__()
        self._devices = parse_device(device)
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._n_epoch = n_epoch
        self._lr = lr
        self._sample_size = sample_size
        self._sample_duration = sample_duration
        self._input_channels = input_channels


    def train(self, df_path, data_root, output_dir, weights=False,
                  col_image='image_path', col_label='label', col_group=None):
        '''
        train
        '''

        import matplotlib
        matplotlib.use('Agg')
        from fastai.vision import Learner
        from fastai.vision import get_transforms, models
        from fastai.vision import accuracy, AUROC
        from fastai.vision import DataBunch, DatasetType
        from fastai.callbacks import SaveModelCallback
        
        data_root = Path(data_root)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True,exist_ok=True)
        model_name = 'scripted_model.zip'

        df = pd.read_csv(df_path)
        num_classes = df['label'].nunique()
        df_train = df[df.dataset.isin(['train'])]
        df_valid = df[df.dataset.isin(['valid'])]
        df_test = df[df.dataset.isin(['test'])]

        sample_size = self._sample_size
        sample_duration = self._sample_duration
        input_channels = self._input_channels
        num_workers = self._num_workers
        batch_size = self._batch_size
        n_epoch = self._n_epoch
        devices = self._devices

        if len(devices) == 0 or devices[0].lower() != 'cpu':
            pin_memory = True
            device_data = devices[0]
        else:
            pin_memory = False
            device_data = None
        
        from vol_dataset import VolumeDataset
        ds_train = VolumeDataset(df_train,data_root,input_channels)
        ds_valid = VolumeDataset(df_valid,data_root,input_channels)
        ds_test = VolumeDataset(df_test,data_root,input_channels)
        data = DataBunch.create(ds_train, ds_valid, test_ds=ds_test, bs=batch_size, 
                                num_workers=num_workers, device=device_data, pin_memory=pin_memory)
        print(df_train.shape, df_valid.shape, df_test.shape)

        from resnet3d import r3d_18 as resnet18
        model = resnet18(input_channels=input_channels,num_classes=num_classes)
        model_single = model
        if len(devices) >= 2:
            model = nn.DataParallel(model_single,device_ids=devices)
        
        if isinstance(weights,bool):
            if weights:
                weights = 1/ds_train.label.value_counts(sort=False)
                weights = weights.values/weights.min()
            else:
                weights = [1,1,1]
        elif isinstance(weights,str) and ',' in weights:
            weights = [float(w) for w in weights.split(',')]
        elif isinstance(weights,list) or isinstance(weights,tuple):
            pass
        weights = torch.tensor(weights)
        loss_func = nn.CrossEntropyLoss(weight=weights)
        loss_func = loss_func.to(devices[0])
        
        metrics = [accuracy]
        metrics += [AUCk(num_classes-1)]
        learn = Learner(data, model, metrics=metrics, wd=1e-2, path=output_dir, loss_func=loss_func)

        lr = self._lr
        learn.fit_one_cycle(n_epoch, slice(lr), callbacks=[SaveModelCallback(learn, every='improvement',monitor='valid_loss', name='best')])
        lr = self._lr/10
        learn.fit_one_cycle(n_epoch, slice(lr), callbacks=[SaveModelCallback(learn, every='improvement',monitor='valid_loss', name='best')])

        x_sample = torch.rand((2,input_channels,sample_duration,sample_size,sample_size))
        x_sample = x_sample.to(devices[0])
        model_scripted = torch.jit.trace(model_single,x_sample)
        model_scripted.to('cpu')
        model_scripted.save(str(output_dir/model_name))


    def infer(self, data_dir, output_path, model_path, df_path=None,
                col_group=None):
        '''
        test
        '''
        from vol_dataset import VolumeDataset

        data_dir = Path(data_dir)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True,exist_ok=True)
        if len(self._devices) > 0:
            device = self._devices[0]
        else:
            device = 'cpu'
        model = torch.jit.load(str(Path(model_path).resolve()),map_location='cpu')
        model.to(device)

        if df_path:
            df = pd.read_csv(df_path)
            if col_group:
                df = df[df[col_group].isin(['valid','test'])].copy()
        else:
            paths = [p.relative_to(data_dir) for p in data_dir.glob('**/*.npy')]
            df = pd.DataFrame({'path':paths, 'label':[-1]*len(paths)})
        df = df.reset_index(drop=True)
        paths = list(df['path'])

        input_channels = self._input_channels
        dataset = VolumeDataset(df,data_dir,input_channels)
        dataloader = DataLoader(dataset, shuffle=False, 
                                batch_size=self._batch_size, num_workers=self._num_workers)

        results = []
        for batch_x, batch_y in tqdm(dataloader):
            batch_x = batch_x.to(device)
            probs = model(batch_x)
            probs = nn.functional.softmax(probs,dim=1)
            probs = probs.to('cpu').data.numpy()
            for path, prob in zip(paths,probs):
                result = {
                    'image_path': str(path),
                    'data_dir': str(data_dir),
                    'pred': prob.argmax(),
                }
                results += [result]
            paths = paths[len(batch_x):]
        df_result = pd.DataFrame(results)
        df_output = df.merge(df_result,left_index=True,right_index=True)
        df_output.to_csv(output_path)

if __name__ == '__main__':
    fire.Fire(VolumeClassification)