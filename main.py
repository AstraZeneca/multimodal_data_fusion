#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# Copyright 2020 by AstraZeneca
# All rights reserved.

@authors: ML&AI - Oncology Data Science
@email: harish.raviprakash@astrazeneca.com
@description: Main file for the multiomics pipeline
"""

import os
import argparse
from src.modeling.pipeline import pipeline_main


def main(args: argparse.ArgumentParser):
    """Interface to the main pipeline

    Args:
        args (argparse.ArgumentParser): Arguments for running the pipeline
    """
    csv_path = args.csv
    config_path = args.yaml
    experiment = args.expt
    pipeline_main(config_path=config_path, csv_path=csv_path, expt_name=experiment)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multiomics survival pipeline")
    parser.add_argument('--yaml', '-y', default="config.yml", type=str, help='Configuration dictionary')
    parser.add_argument('--csv', '-c', default='', 
                        type=str, help='Full path to the features csv')
    parser.add_argument('--expt', '-e', default='000001', 
                        type=str, help='Experiment number')
    args = parser.parse_args()
    main(args)