# export
from ast import Pass
import shutil
import os
import sys
import random
import yaml
import numpy as np
import dataclasses
import argparse
import time

import numpy as np
import torch

from pprint import pprint
from pathlib import Path
import torch


def set_seed_all(seed):
    """
    Sets all random states
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def set_cudnn_flags():
    """Set CuDNN flags for reproducibility"""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device():
    """Returns torch.device"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


class Params:
    """
    A class to help with experiment parameter handling
    """
    def __init__(self):
        pass
        # Following attributes needed for get_exp_savepath, and fields_used_in_exp_dirname methods
        self.fields_used_in_exp_dirname = ["batch_size"]
        self.fieldnames_used_in_exp_dirname = True
        self.results_dir : str =  "./changeme/" # dir where the experiment is saved

    def as_dict(self):
        param_dict = {}
        for k in self.__dict__.keys():
            if "-" in k: print(f"Please replace the '-'s in {k} with '_'s")
            if k.startswith('__'): continue
            elif type(self.__dict__[k]) == classmethod: continue # We don't want methods like this one
            else: param_dict[k] = self.__dict__[k]
        return param_dict

    def get_dirname_params_dict(self):
        return {key:val for key, val in self.as_dict().items() if key in self.fields_used_in_exp_dirname}

    def get_exp_savepath(self):
        s = ''
        for key in sorted(self.get_dirname_params_dict().keys()): # don't use selected_param_dict, as we might (cosmetically) care about order
            if self.fieldnames_used_in_exp_dirname:
                s += f'{key}-{self.as_dict()[key]}_'
            else:
                s += f'{self.as_dict()[key]}_'

        exp_dirname = Path(s[:-1])
        assert len(str(exp_dirname)) < 255 # max folder_name length
        results_dir = Path(self.results_dir)
        exp_name = Path(f'seed-{self.seed}')

        return results_dir/exp_dirname/exp_name

    def write_to_config_yaml(self, out_dir=None, fname="exp_config.yaml"):
        """
        Writes a yaml config file (fname) to outdir. Defaults to
        that returned by get_exp_savepath() if none provided
        """
        if not self.get_exp_savepath().exists():
            self.get_exp_savepath().mkdir(parents=True, exist_ok=True)

        config_filepath = self.get_exp_savepath()/fname
        with open(config_filepath, 'w') as stream:
            yaml.dump(self.as_dict(), stream, default_flow_style=False)

        print(f'Wrote params as config file: \n',config_filepath)

    def set_from_config_yaml(self, config_path):
        with open(config_path, 'r') as stream:
            conf_dict = yaml.load(stream, Loader=yaml.FullLoader)
            self.set_from_config_dict(conf_dict)
        print("Loaded config file: \n", conf_dict)

    def set_from_config_dict(self, config_dict):
        for key, item in config_dict.items():
                self.__dict__[key] = item

    def __repr__(self):
        s = ''
        s += 'Experiment parameters \n'
        self_dict = self.as_dict()
        for key in sorted(self_dict.keys(), key=str.casefold):
            if key == 'selected_params':continue
            s += f' - {key} : {self_dict[key]} \n'
        return s

def write_params_as_wandb_yaml(flags, out_dir=None, fname="params.yaml"):
    """
    Writes a yaml config file (fname) to outdir. 
    Defaults current working directory if out_dir=none
    
    Todo: convert dictionary entries, e.g "value: []" to 
    be compatibile with wandb
    """
    pass

def convert_params_dict_to_wandb(pdict):
    wandb_dict = {}
    ignore_keys = ["fields_used_in_exp_dirname", "fieldnames_used_in_exp_dirname"]
    param_lens = []
    for key, val in pdict.items():
        if key in ignore_keys:
            continue
        if type(val) is not dict:
            if type(val) is list or type(val) is tuple:
                wandb_val = {"values":val}
                param_lens.append(len(val))
            else:
                wandb_val = {"value":val}
            wandb_dict[key] = wandb_val
        else:
            wandb_dict[key] = val
    #print(f" There were {np.prod(param_lens)} parameter combinations ")
    return wandb_dict, np.prod(param_lens)
        #wandb_dict[key]
    
    # function to iterate


# export
if __name__ == "__main__":
    # In order for imports to work correctly:
    # run in the shell 'export PYTHONPATH=$PYTHONPATH:~/dann-rnns'
    pass

