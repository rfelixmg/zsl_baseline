import os, sys
from docutils.nodes import copyright
from jinja2.utils import object_type_repr

sys.path.append(os.path.abspath('..'))

import numpy as np
import h5py
import json
import argparse, random
from utils import file_utils


base = '/home/rfelixmg/Dropbox/PROJETOS/ZSL_DS_SJE/experiments/cub/'
file_names = file_utils.get_files(base, 'txt')
format_result = []

for file_name in file_names:
    result = None
    with open(base + file_name, 'r') as in_:
        result = json.load(in_)

    evaluation = result['evaluation']
    configuration = result['configuration']

    line = [str(result['configuration']['output_file'].split('/')[-1][:-4]),
            result['evaluation']['accuracy_test'],
            result['evaluation']['coeficient_determination_test'],
            result['evaluation']['precision_test'],
            result['evaluation']['recall_test'],

            result['evaluation']['accuracy_train'],
            result['evaluation']['coeficient_determination_train'],
            result['evaluation']['precision_train'],
            result['evaluation']['recall_train']
            ]

    format_result.append(line)
file_utils.save_list_to_list_txt(format_result, 'my_obj', 'txt', './')