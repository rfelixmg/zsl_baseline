import os, sys
from docutils.nodes import copyright
from jinja2.utils import object_type_repr

sys.path.append(os.path.abspath('..'))

import numpy as np
import h5py
import argparse, random
from utils import file_utils, text_utils

def bow(text):
    """
        bow: This method aims to create a bag-of-word representation for a given text.

        parameters:
        ----------
        text: string - Raw-Text to be represented

    """
    unique_words = np.unique(text)

    word_to_id = {}
    id_to_word = {}
    for key, word in enumerate(unique_words):
        word_to_id[word] = key
        id_to_word[key] = word

    return word_to_id, id_to_word

def convert_bow(sample, vocabulary):

    binary_vector = np.zeros((vocabulary.__len__()))

    sample = text_utils.clean_str(sample)
    sample = text_utils.remove_punctuation(sample)

    for word in sample.split():

        binary_vector[vocabulary[word]] = 1

    return binary_vector

def save_h5py(bow_dataset, directory):

    n_total = np.size(bow_dataset.keys())
    for n_key, n_class in enumerate(bow_dataset.keys()):

        f = h5py.File(directory + n_class + '.h5', 'w')
        fgroup = f.create_group(n_class)

        sys.stdout.write('\rSaving class: %s (%d/%d)' % (n_class, n_key, n_total ))
        sys.stdout.flush()

        for img_name in bow_dataset[n_class].keys():
            binary_vector = np.array(bow_dataset[n_class][img_name]).astype(np.int)
            fgroup.create_dataset(img_name, data=binary_vector)

        f.close()
    print '\n'

def collect_dataset(directory):

    raw_text = ''
    raw_dataset = {}
    for folder in file_utils.get_folders(directory):
        folder_path = directory + folder + '/'
        new_samples = {}
        for txt_file in file_utils.get_files(folder_path, obj_type='txt'):
            t, lines = file_utils.load_text(txt_file, folder_path)
            new_samples[txt_file[:-4]] = lines
            raw_text += ' ' + t

        raw_dataset[folder] = new_samples



    return raw_text, raw_dataset



def main():
    # Getting arg parameters
    parser = argparse.ArgumentParser(description='Processing dataset to generate BoW representation')
    parser.add_argument('-d', help='Textual dataset folder', required=False)
    parser.add_argument('-o', help='Output folder', required=False)
    parser.add_argument('-n', help='Output folder name')
    parser.add_argument('-verbose', help='Debug mode', required=False)

    args = vars(parser.parse_args())

    dataset_folder= str(args['d']) if args['d'] else '../data/oxford_flowers/fine_grained_description/'
    output_folder= str(args['o']) if args['o'] else '../data/oxford_flowers/features/'
    output_name= str(args['n']) if args['n'] else 'bow'

    print '\n\n\n'
    print '-'*50
    print 'Dataset_folder: ', dataset_folder
    print 'Output_Folder: ', output_folder
    print 'Folder: ', output_name
    print '\n\n\n'



    # Generate vocabulary for BoW
    raw_text, raw_dataset = collect_dataset(dataset_folder)
    text = text_utils.clean_str(raw_text)
    text = text_utils.remove_punctuation(text)
    text = text.split()
    vocabulary, id_to_word = bow(text)

    bow_dataset = raw_dataset.copy()
    # Generate BoW representation

    for n_class in raw_dataset.keys():
        for img_name in raw_dataset[n_class].keys():
            for key, sample in enumerate(raw_dataset[n_class][img_name]):
                bow_dataset[n_class][img_name][key] = convert_bow(sample, vocabulary)

    file_utils.makedir(output_name, output_folder)
    save_h5py(bow_dataset, output_folder + output_name + '/')

    return vocabulary, id_to_word, raw_dataset, bow_dataset

if __name__ == '__main__':
    random.seed(0)
    vocabulary, id_to_word, raw_dataset, bow_dataset = main()

    print 'Dataset create with sucess'