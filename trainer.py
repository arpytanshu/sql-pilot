
import os
import sys
import logging
import numpy as np
from pathlib import Path    
from typing import Union
from time import perf_counter
from datetime import datetime
from dataclasses import dataclass
from collections import defaultdict

import fire
import torch
from torch.optim import AdamW

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR

from model.VAE import VAEConfig, VAEModel, VAELoss
from data.celeb_utils import get_dataset


@dataclass
class TrainingArgs:
    dataset_path: Path = None
    device: torch.device = torch.device('cuda:0')
    dtype: torch.dtype = torch.float32
    chkpt_base_dir: Path = Path('runs/')
    
    train_batch_sz: int = 384
    test_batch_sz: int = 640

    beta1: float = 0.9
    beta2: float = 0.95
    learning_rate: float = 1e-4
    min_lr: float = 1e-6
    kld_weight: float = 0.0001
    
    max_iters: int = 5001

    model_cfg: VAEConfig = VAEConfig()

    curr_iter: int = 1
    log_interval: int = 25
    eval_iterval: int = 150
    plot_interval: int = 50
    best_model_stat: float = None
    worse_report_counter: int = 0

    save_plots: bool = True
    n_samples_visualize: int = 3072


class Trainer:
    def __init__(self, args: TrainingArgs = None, 
                 checkpoint_path: Union[Path , str] = None,
                 override_args: dict = {}):
        '''
        Either one of `args` or `checkpoint_path` must be provided.
        
        To initiate a new run, provide `args`.
        To resume from checkpoint, provide `checkpoint_path`.

        If both are provided, will resume from `checkpoint_path` by default.

        To override arguments when resuming from checkpoints, 
            provide a dictionary of arguments in `override_args`.
        '''
        self.chkpt_file_name = "model.th"
        
        if checkpoint_path is not None:
            if self.checkpoint_exists(checkpoint_path):
                self.checkpoint_path = Path(checkpoint_path)
                self.load_checkpoint()
                self.logger = self.create_logger()
                self.logger.info(f"Successfully loaded checkpoint from iter:{self.args.curr_iter}")
                self._override_args(override_args)
            else:
                raise ValueError('Checkpoint not present at provided {checkpoint_path=}')
        
        elif args is not None:
            self.args = args
            self.checkpoint_path = self.create_checkpoint_dir()
            
            self.logger = self.create_logger()
            self.logger.info('checkpoint_path not provided. starting new run.')
            self.logger.info('dumping args...')
            self.logger.info(str(self.args.__dict__))

            self.model = self.init_model()
            self.optimizer = self.get_optimizer()
            self.scheduler = self.get_scheduler()
        else:
            raise ValueError('Expected either args or checkpoint_path')

        self.tr_dataloader = self.get_train_dataloader()
        self.te_dataloader = self.get_test_dataloader()
        self.criterion = self.get_loss_fn()
        self.logger.info("All initializations complete.")

        self.writer = SummaryWriter(self.checkpoint_path)

    def train(self):
        self.logger.info(f"Resuming training from iter={self.args.curr_iter}")
        for itern in range(self.args.curr_iter, self.args.max_iters):
            self.args.curr_iter = itern
            
            images, _ = next(iter(self.tr_dataloader))
            
            self.model.train()
            self.optimizer.zero_grad()
            
            images = images.to(self.args.dtype).to(self.args.device)
            # labels = labels.to(self.args.dtype).to(self.args.device)

            reconst, _, mu, log_var = self.model(images)

            losses = self.compute_loss(input = images, 
                                       reconst = reconst, 
                                       mu = mu, 
                                       log_var=log_var)

            losses['loss'].backward()
            self.optimizer.step()
            self.scheduler.step()

            tr_rpt = dict(
                train_loss = round(losses['loss'].detach().item(), 4),
                train_recon_loss = round(losses['recon_loss'].item(), 4),
                train_kld = round(losses['kld'].item(), 4),
                learning_rate=round(self.optimizer.param_groups[0]['lr'], 6))

            self.write_tensorboard(tr_rpt)
            
            # log training stuff
            if (itern % self.args.log_interval) == 0:
                self.logger.info(f"{itern=} tr_loss:{tr_rpt['train_loss']} ")
                self.logger.info(f"tr_recon_loss:{tr_rpt['train_recon_loss']} tr_kld:{tr_rpt['train_kld']}")
            
            # log evaluation stuff + checkpoint
            if (itern % self.args.eval_iterval) == 0:
                self.logger.info(f"Running evaluation at {itern=}...")
                te_rpt = self.evaluate()
                self.write_tensorboard(te_rpt)
                self.logger.info(f"{itern=} te_loss:{te_rpt['test_loss']} elapsed:{te_rpt['elapsed']}")
                self.logger.info(f"te_recon_loss:{te_rpt['test_recon_loss']} te_kld:{te_rpt['test_kld']}")
                self.checkpoint_logic(te_rpt, tr_rpt)

            # plot stuff
            elif (self.args.save_plots) and ((itern % self.args.plot_interval) == 0):
                self.model.get_latent_space_plot(self.te_dataloader, 
                                                    self.args.n_samples_visualize, 
                                                    self.args, 
                                                    save_plot=True)


    def checkpoint_logic(self, test_report, train_report):
        '''
        makes decision for:
        - checkpoint model at current iteration
        - skip checkpointing
        - stop training.
        using the current evaluation report.
        '''
        if self.args.best_model_stat == None:
            self.logger.info('Updated best model. best_model_stat empty.')
            self.args.best_model_stat = test_report['test_loss']
            self.args.worse_report_counter = 0
            self.save_checkpoint()
        elif (self.args.best_model_stat > test_report['test_loss']) and \
            (test_report['test_loss'] < train_report['train_loss']):
            self.logger.info('Updating best model.')
            self.logger.info('loss improved from {} to {}'.format(
                self.args.best_model_stat, test_report['test_loss']))
            self.args.best_model_stat = test_report['test_loss']
            self.args.worse_report_counter = 0
            self.save_checkpoint()
        else:
            self.args.worse_report_counter += 1
            self.logger.info(F"Updated {self.args.worse_report_counter=}")
        
        if self.args.worse_report_counter > 5:
            self.logger.info(F"evaluation worse over multiple successive runs.")
            self.logger.info(F"Abort Training at {self.args.curr_iter=}")
            sys.exit()

    def save_checkpoint(self):
        chkpt = dict(
            args = self.args,
            model = self.model.state_dict(),
            optimizer = self.optimizer.state_dict(),
            scheduler = self.scheduler.state_dict()
        )
        filename = self.checkpoint_path / self.chkpt_file_name
        torch.save(chkpt, filename)
        self.logger.info(f"Saved checkpoint at {filename}")

    def load_checkpoint(self, ):
        chkpt = torch.load(self.checkpoint_path / self.chkpt_file_name)
        self.args = chkpt['args']
        self.model = VAEModel(self.args.model_cfg)
        self.model.load_state_dict(chkpt['model'])
        self.model.to(self.args.dtype).to(self.args.device)

        self.optimizer = self.get_optimizer()
        self.optimizer.load_state_dict(chkpt['optimizer'])

        self.scheduler = self.get_scheduler()
        self.scheduler.load_state_dict(chkpt['scheduler'])
        
    def evaluate(self):
        '''
        Runs an evaluation loop over the provided dataloader
        Returns a dictionary of metrics.
        '''
        self.model.eval()
        result_dd = defaultdict(list)
        
        tick = perf_counter()
        for images, _ in self.te_dataloader:
            images = images.to(self.args.dtype).to(self.args.device)
            with torch.no_grad():
                reconst, z, mu, log_var  = self.model(images)
                losses = self.compute_loss(input = images, 
                                           reconst = reconst, 
                                           mu = mu, 
                                           log_var=log_var)
        
                result_dd['test_loss'].append(losses['loss'].detach().item())
                result_dd['test_recon_loss'].append(losses['recon_loss'].item())
                result_dd['test_kld'].append(losses['kld'].item())
        elapsed = perf_counter() - tick

        result = dict()
        for key in result_dd.keys():
            value = sum(result_dd[key]) / len(result_dd[key])
            result[key] = round(value, 3)
        result['elapsed'] = round(elapsed, 3)

        return result

    def get_scheduler(self):
        return CosineAnnealingLR(self.optimizer, 
                              T_max = self.args.max_iters,
                              eta_min = self.args.min_lr,
                              last_epoch = -1
                              )
    
    def get_optimizer(self):
        return AdamW(self.model.parameters(),
                     lr=self.args.learning_rate,
                     betas=(self.args.beta1, self.args.beta2)
                     )

    def checkpoint_exists(self, path: Union[Path , str]):
        chkpt_file = Path(path) / self.chkpt_file_name
        return os.path.exists(chkpt_file)

    def _override_args(self, override_args: dict):
        # inplace update self.args with entries in override_args/
        self.args.__dict__.update(override_args)
      
    def get_train_dataloader(self):
        tr_dataset = get_dataset(self.args.dataset_path, split='train')
        tr_dataloader = DataLoader(tr_dataset, 
                                   batch_size=self.args.train_batch_sz, 
                                   shuffle=True)
        return tr_dataloader

    def get_test_dataloader(self):
        te_dataset = get_dataset(self.args.dataset_path, split='test')
        te_dataloader = DataLoader(te_dataset,
                                   batch_size=self.args.test_batch_sz, 
                                   shuffle=False)
        return te_dataloader

    def get_loss_fn(self):
        return VAELoss

    def init_model(self):
        return VAEModel(self.args.model_cfg).\
            to(self.args.dtype).\
            to(self.args.device)

    def compute_loss(self, input, reconst, mu, log_var):
        return self.criterion(input=input, 
                              reconst=reconst, 
                              mu=mu, 
                              log_var=log_var,
                              kld_weight=self.args.kld_weight)
    
    def create_checkpoint_dir(self):
        # create a dir w/ current date-time inside the base dir.
        chkpt_name = datetime.now().strftime("%y%m%d-%H%M")
        chkpt_name = Path(self.args.chkpt_base_dir) / chkpt_name
        os.makedirs(chkpt_name, exist_ok=True)
        # create a plots dir inside the base dir
        plots_dir = chkpt_name / "plots"
        os.makedirs(plots_dir, exist_ok=True)
        return chkpt_name
    
    def create_logger(self):
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

        # Create handlers
        c_handler = logging.StreamHandler()
        f_handler = logging.FileHandler(self.checkpoint_path / 'logs.log')
        c_handler.setLevel(logging.INFO)
        f_handler.setLevel(logging.INFO)
        
        # Create formatters and add it to handlers
        c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
        f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        c_handler.setFormatter(c_format)
        f_handler.setFormatter(f_format)
        
        # Add handlers to the logger
        logger.addHandler(c_handler)
        logger.addHandler(f_handler)

        return logger

    def write_tensorboard(self, report: dict):
        for key in report.keys():
            self.writer.add_scalar(key, report[key], self.args.curr_iter)
    


            
def main(    
        dataset_path,
        checkpoint_path = None,
        train = True,
        eval = True,
        device = 'cuda',
        dtype = 'fp32', # fp32, bf16,
        chkpt_base_dir = 'runs/',
        train_batch_sz = 384,
        test_batch_sz = 2048,
        max_iters = 10001,
        log_interval = 25,
        eval_iterval = 150,
        plot_interval = 50
):
    if dtype == 'fp32':
        dtype = torch.float32
    elif dtype == 'bf16':
        dtype = torch.bfloat16
    else:
        raise ValueError('Expected type to be one of `fp32` or `bf16`.')
    
    checkpoint_path = checkpoint_path
    if checkpoint_path is not None:
        override_args=dict(
            dataset_path = dataset_path,
            device = torch.device(device),
            dtype = dtype,
            chkpt_base_dir = Path(chkpt_base_dir),
            train_batch_sz = train_batch_sz,
            test_batch_sz = test_batch_sz,
            max_iters = max_iters,
            log_interval = log_interval,
            eval_iterval = eval_iterval,
            plot_interval = plot_interval
            )
        trainer = Trainer(checkpoint_path=checkpoint_path,
                          override_args=override_args)
    else:
        args = TrainingArgs()
        args.dataset_path = Path(dataset_path)
        args.device = torch.device(device)
        args.dtype = dtype
        args.chkpt_base_dir = Path(chkpt_base_dir)
        args.train_batch_sz = train_batch_sz
        args.test_batch_sz = test_batch_sz
        args.max_iters = max_iters
        args.log_interval = log_interval
        args.eval_iterval = eval_iterval
        args.plot_interval = plot_interval

        trainer = Trainer(args)
    
    if train:
        trainer.train()
    if eval:
        trainer.evaluate()


if __name__ == '__main__':
    fire.Fire(main)