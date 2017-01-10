import os, sys
from docutils.nodes import copyright
from jinja2.utils import object_type_repr

sys.path.append(os.path.abspath('..'))

import numpy as np
import h5py
import argparse, random
from utils import file_utils, text_utils

def savefile_h5py(binary_vector, directory):

    f = h5py.File(directory + 'attributes.h5', 'w')
    fgroup = f.create_group('attributes')

    fgroup.create_dataset('attributes', data=binary_vector)

    f.close()
    print '\n'

def collect_dataset(data_file):


    raw_data = np.loadtxt(data_file)
    binary_vector = np.zeros((11788, 312))

    for line in raw_data:
        img_id = int(line[0]) - 1
        att_id = int(line[1]) - 1
        is_present = int(line[2])
        binary_vector[img_id][att_id] = is_present

    return binary_vector

def process_awa_attributes(data_file):

    return None


def main():
    # Getting arg parameters
    parser = argparse.ArgumentParser(description='Processing attributes to generate binary vector representation')
    parser.add_argument('-d', '-dataset',
                        default='../data/cub/attributes/image_attribute_labels.txt',
                        help='Textual dataset file', required=False)
    parser.add_argument('-o', '-output',
                        default='../data/cub/features/',
                        help='Output folder', required=False)
    parser.add_argument('-n', '-name',
                        default='bow_attributes',
                        help='Output folder name')
    parser.add_argument('-verbose', help='Debug mode', required=False)

    args = vars(parser.parse_args())
    dataset_file = str(args['d'])
    output_folder= str(args['o'])
    output_name= str(args['n'])

    print '\n\n\n'
    print '-'*50
    print 'Dataset_folder: ', dataset_file
    print 'Output_Folder: ', output_folder
    print 'Folder: ', output_name
    print '\n\n\n'


    # Generate vocabulary for BoW
    att_vector = collect_dataset(dataset_file)
    file_utils.makedir(output_name, output_folder)

    savefile_h5py(att_vector, output_folder + output_name + '/')

    return att_vector

if __name__ == '__main__':
    random.seed(0)
    att_vector = main()

    print 'Dataset create with sucess'