#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# Copyright 2020 by AstraZeneca
# All rights reserved.

@timestamp: Tue July 20 09:23:18 2021
@author: Harish RaviPrakash
@email: harish.raviprakash@astrazeneca.com
@description: Script to save config dictionaries to yaml and load from yaml
"""

import os, sys, glob
import yaml
from deepdiff import DeepDiff
from typing import Union, Dict


def save_config(expt_config: Dict[str, Union[int, float, bool, list]], expt_name: str, savepath: str) -> None:
    """
    Method to save the experiment config

    Parameters
    ----------
    expt_config : DICT, required
        dicitonary of configuration parameters
    expt_name : STRING, required
        experiment name to save and check for existing versions
    savepath : STRING, required
        location to store the config yaml
    
    Returns
    -------
    None
    """
    # subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip().decode("utf-8")
    is_new = True
    # Check for existing versions of expt_name
    cnt = 0
    for config_name in glob.glob(os.path.join(savepath, expt_name + "*.yml")):
        cnt += 1
        with open(config_name, 'r') as config_file: 
            config = yaml.safe_load(config_file)
        if DeepDiff(expt_config, config, ignore_order=True) == {}:
            is_new = False
    
    if is_new:
        filename = ''
        if cnt:
            filename = expt_name + '_v' + str(cnt) + '.yml'
        else:
            filename = expt_name + '.yml'
        with open(os.path.join(savepath, filename), 'w') as yml_file:
            yaml.dump(expt_config, yml_file, default_flow_style=False)

    return


def load_config(expt_name: str, loadpath: str, version: int = -1) -> Dict[str, Union[int, float, bool, list]]:
    """
    Method to load a saved config file

    Parameters
    ----------
    expt_name : STRING, required
        experiment name
    loadpath : STRING, required
        location where config files are stored
    version : INTEGER, optional
        specific version to be loaded. If omitted the latest version is loaded
    
    Returns
    -------
    expt_config : DICT
        configuration dictionary
    """
    if version == -1:
        config_files = [config_fname for config_fname in glob.glob(os.path.join(loadpath, expt_name + '_v*.yml'))]
        if len(config_files) == 0:
            raise FileNotFoundError("{} not found. Check paths and experiment number and try again.".format(expt_name))
        with open(os.path.join(loadpath, expt_name + '_v' + str(len(config_files) - 1) + '.yml'), 'r') as config_file:
            expt_config = yaml.safe_load(config_file)
        return expt_config
    else:
        if not os.path.isfile(os.path.join(loadpath, expt_name + '_v' + str(version))):
            raise FileNotFoundError("{} not found. Check paths and versions and try again.".format(expt_name + '_v' + str(version)))
        with open(os.path.join(loadpath, expt_name + '_v' + str(version) + '.yml'), 'r') as config_file:
            expt_config = yaml.safe_load(config_file)
        return expt_config