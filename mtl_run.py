import os
import sys
import time
import argparse
import importlib
import collections
import numpy as np
import multiprocessing


from utils.configure import PDConfig
from utils.configure import JsonConfig, ArgumentGroup, print_arguments
from backbone.bert_model import BertModel
backbones = {
    "bert_model": BertModel
}
sys.path.append("reader")
sys.path.append("optimizer")
sys.path.append("paradigm")
sys.path.append("backbone")
TASKSET_PATH="config"


def train(multitask_config): 
    print("Loading multi_task configure...................")
    args = PDConfig(yaml_file=[multitask_config])
    args.build()
    
    index = 0
    reader_map_task = dict()
    task_args_list = list()
    reader_args_list = list()
    id_map_task = {index: args.main_task}
    print("Loading main task configure....................")
    main_task_name = args.main_task
    task_config_files = [i for i in os.listdir(TASKSET_PATH) if i.endswith('.yaml')]
    main_config_list = [config for config in task_config_files if config.split('.')[0] == main_task_name]
    main_args = None
    for config in main_config_list: 
        main_yaml = os.path.join(TASKSET_PATH, config)
        main_args = PDConfig(yaml_file=[multitask_config, main_yaml])
        main_args.build()
        main_args.Print()
        if not task_args_list or main_task_name != task_args_list[-1][0]: 
            task_args_list.append((main_task_name, main_args))
        reader_args_list.append((config.strip('.yaml'), main_args))
        reader_map_task[config.strip('.yaml')] = main_task_name

    print("Loading auxiliary tasks configure...................")
    aux_task_name_list = args.auxiliary_task.strip().split()
    for aux_task_name in aux_task_name_list: 
        index += 1
        id_map_task[index] = aux_task_name
        print("Loading %s auxiliary tasks configure......." % aux_task_name)
        aux_config_list = [config for config in task_config_files if config.split('.')[0] == aux_task_name]
        for aux_yaml in aux_config_list: 
            aux_yaml = os.path.join(TASKSET_PATH, aux_yaml)
            aux_args = PDConfig(yaml_file=[multitask_config, aux_yaml])
            aux_args.build()
            aux_args.Print()
            if aux_task_name != task_args_list[-1][0]: 
                task_args_list.append((aux_task_name, aux_args))
            reader_args_list.append((aux_yaml.strip('.yaml'), aux_args))
            reader_map_task[aux_yaml.strip('.yaml')] = aux_task_name
 # import tasks reader module and build joint_input_shape
    input_shape_list = []
    reader_module_dict = {}
    input_shape_dict = {}


if __name__ == '__main__':

    multitask_config = "mtl_config.yaml"
    train(multitask_config)