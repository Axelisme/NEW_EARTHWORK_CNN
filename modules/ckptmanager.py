"""
define a class to manage the checkpoint
"""
import os
import yaml
from pathlib import Path
from typing import Optional, Dict

import torch
from torch.nn import Module

from util.io import show


class CheckPointManager:
    def __init__(self,
                 ckpt_dir:str,
                 keep_num:int = 3,
                 **kwargs):
        self.ckpt_dir   = ckpt_dir

        assert keep_num > 0, "keep_num should be greater than 0"
        self.keep_num   = keep_num

    def save_config(self, save_conf:Dict, config_name:str = 'config.yaml'):
        """Save the config file."""
        dir_path = Path(self.ckpt_dir)
        dir_path.mkdir(parents=True, exist_ok=True)
        config_path = Path(self.ckpt_dir) / config_name
        with open(config_path, 'x') as f:
            yaml.dump(save_conf, f, default_flow_style=False)


    def load_model(self,
            model:Module,
            ckpt_path: str):
        """Load checkpoint and return save config if exists.
           Notice: only overwrite the model, optimizer and scheduler, not the config."""
        # create the default path if not specified
        ckpt_path:Path = Path(ckpt_path)

        # check if the file exists
        if not ckpt_path.exists():
            raise FileNotFoundError(f"File {ckpt_path} does not exist.")

        # load checkpoint
        show(f'Loading checkpoint from {ckpt_path}')
        checkpoint = torch.load(ckpt_path, map_location='cpu')

        model.load_state_dict(checkpoint['model'])

        # return the save config
        return checkpoint.get('meta')


    def save_model(self,
            model:Module,
            ckpt_name: str,
            meta_conf: Dict,
            overwrite: bool = False):
        """Save the checkpoint."""
        # create the default path if not specified
        ckpt_path = Path(self.ckpt_dir) / ckpt_name

        # check if the file exists and create the directory if not
        if ckpt_path.exists() and not overwrite:
            raise FileExistsError(f"File {ckpt_path} already exists.")
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)

        # save the checkpoint
        show(f'Saving model to {ckpt_path}')
        save_dict = {
            'model': model.state_dict(),
            'meta': meta_conf
        }
        torch.save(save_dict, ckpt_path)

        # clean old checkpoints
        ckpts = list(Path(self.ckpt_dir).glob(f"dev-step-*.pth"))
        ckpts = sorted(ckpts, key=lambda ckpt: int(Path(ckpt).stem.split('-')[-1]))
        if len(ckpts) > self.keep_num:
            for ckpt in ckpts[:-self.keep_num]:
                os.remove(ckpt)
