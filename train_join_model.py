#!/usr/bin/env python

"""

"""
from sklearn.preprocessing import label

__author__ = "Rafael Felix"
__license__ = "UoA"
__copyright__ = "Copyright 2016"
__maintainer__ = "Rafael Felix"
__email__ = "rfelixmg@gmail.com"
__status__ = "beta"

import random, os, sys, argparse
from _ast import Lambda

import json

import numpy as np
import h5py
import torchfile
import time
from utils.roc import roc

import matplotlib.pyplot as plt

from utils import file_utils

#from keras.models import Model, Sequential
#from keras.layers import Merge, Flatten, Input, Dense

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib

from sklearn.svm import SVC, LinearSVC
from sklearn.multiclass import OneVsRestClassifier
import sklearn.metrics as metrics

import modules.bow_encoder as bow
import modules.googlenet_encoder as googlenet

sys.path.append(os.path.abspath('..'))

tag_ = 'awa'
dataset_ = './data/%s/' % tag_

configuration = {
    'dataset':                  dataset_,
    'dataset_image':            dataset_ + 'images/',
    'dataset_text':             dataset_ + 'fine_grained_description/',
    'dataset_attributes':       dataset_ + 'attributes/',
    'embedding':                dataset_ + 'features/',
    #'embedding_image':          dataset_ + 'features/halah_googlenet/feature.txt',
    'embedding_image':          dataset_ + 'features/lampert_vgg/feature.h5',
    'embedding_text':           dataset_ + 'features/bow_text/None',
    'embedding_attributes':     dataset_ + 'features/bow_attributes/feature.txt',
    'estimation_attributes':    dataset_ + '/attributes/class_attribute_labels_continuous.txt',
    #   rfr, svm, linear, logistic, linear-svm
    '#classes':                 None,
    'baseline_model':           'linear-svm',
    '#neighbors':               1,
    'number_epochs':            10,
    'n_estimators':             100,
    'max_iter':                 200,
    'n_jobs':                   -2,
    'estimated_values':         False,
    'output_file':              '',
    'tag':                      tag_,
    'C':                        10.
}

configuration['output_file'] = './experiments/%s/%s_evaluation_%dnn%s.txt' % \
                                (configuration['tag'],
                                 configuration['baseline_model'],
                                 configuration['#neighbors'],
                                 '_estimated_values' if configuration['estimated_values'] else '')

if configuration['tag'] == 'cub':
    configuration['#classes'] = 200
elif configuration['tag'] == 'awa':
    configuration['#classes'] = 50


evaluation = {}

def plot_roc(matrix_results, id_class):
    plt.clf()
    fig = plt.figure(figsize=(30, 15))
    plt.plot([0, 1], [0, 1], 'k--')

    keys = matrix_results.keys()
    keys.sort()
    for key in keys:
        tpr_ = matrix_results[key]['tpr']
        fpr_ = matrix_results[key]['fpr']
        auc_ = matrix_results[key]['auc']

        plt.plot(fpr_, tpr_, label='%s (%d)[auc: %.2f]' % (id_class[str(key + 1)], key + 1, auc_))

    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.show()
    fig.savefig('./experiments/awa/%s_%s_roc_curve.png' % (configuration['baseline_model'], configuration['tag']))

def train_svm(X,y):

    clf = []

    for key in np.arange(y.shape[1]):
        cl_ = SVC(C=10, verbose=10, cache_size=100, max_iter=configuration['max_iter'])
        #cl_ = LinearSVC(C=10, verbose=10, cache_size=100, max_iter=configuration['max_iter'])
        cl_.fit(X, y[:,key])
        clf.append(cl_)

    return clf

def train_linear_svm(X,y):

    clf = []
    t_total = time.time()
    for key in np.arange(y.shape[1]):
        #cl_ = SVC(C=10, verbose=10, cache_size=100, max_iter=configuration['max_iter'])
        t = time.time()
        cl_ = LinearSVC(C=configuration['C'], verbose=10)
        cl_.fit(X, y[:,key])
        print 'Time svm(%d): %f ' %(key, time.time() - t)
        clf.append(cl_)

    print 'Total time: ', time.time() - t_total

    return clf

def predict_svm(clf, X, y):

    prediction = np.zeros(y.shape)
    acc = np.zeros((y.shape[1]))

    print 'Att |     Acc |     Time (s)'
    for key, cl_ in enumerate(clf):
        t = time.time()
        p_ = cl_.predict(X)
        prediction[:,key] = p_
        acc[key] = metrics.accuracy_score(y[:, key], p_)
        print '%d| %f| %f ' %( key, acc[key], (time.time() - t))

    print 'Acc overall: ', metrics.accuracy_score(y, prediction)
    print 'Similarity: ', (y == prediction).sum()/float(y.size)
    return prediction, acc

def collect_splits(directory, vslice=0.1):

    id_labels = np.loadtxt(directory + 'id_labels.txt')[:, 1].astype(np.int) - 1
    labels = np.unique(id_labels)

    try:
        id_train_samples = np.loadtxt(directory + 'train_ids.txt').astype(np.int) - 1
        id_test_samples = np.loadtxt(directory + 'test_ids.txt').astype(np.int) - 1
        id_validation_samples = np.loadtxt(directory + 'validation_ids.txt').astype(np.int) - 1
    except:

        # Load dataset splits
        set_train_classes = np.loadtxt(directory + 'train_classes.txt').astype(np.int) - 1
        set_test_classes = np.loadtxt(directory + 'test_classes.txt').astype(np.int) - 1

        id_train_samples = np.array((), dtype=np.int)
        id_test_samples = np.array((), dtype=np.int)
        id_validation_samples = np.array((), dtype=np.int)

        for label in set_train_classes:
            all_pos = np.where(id_labels == label)[0]
            num_validation = int(all_pos.shape[0] * vslice)
            id_validation_samples = np.concatenate((id_validation_samples, all_pos[:num_validation]))
            id_train_samples = np.concatenate((id_train_samples, all_pos[num_validation:]))

        for label in set_test_classes:
            all_pos = np.where(id_labels == label)[0]
            num_validation = int(all_pos.shape[0] * vslice)
            id_validation_samples = np.concatenate((id_validation_samples, all_pos[:num_validation]))
            id_test_samples = np.concatenate((id_test_samples, all_pos[num_validation:]))

        id_train_samples = np.random.permutation(id_train_samples)

        np.savetxt(directory + 'train_ids.txt', id_train_samples+1, fmt='%d')
        np.savetxt(directory + 'validation_ids.txt', id_validation_samples+1, fmt='%d')
        np.savetxt(directory + 'test_ids.txt', id_test_samples+1, fmt='%d')


    return labels, id_labels, id_train_samples, id_validation_samples, id_test_samples

def load_args():
    # Getting arg parameters
    parser = argparse.ArgumentParser(description='Training LSTM to generate text in char-rnn level')
    parser.add_argument('-d', '-dataset',
                        default=str(configuration['dataset']),
                        help='Textual dataset folder', required=False)

    parser.add_argument('-emb_dim', '-embedding_dimension',
                        help='Dimension of embedding for textual and visual representation',
                        default=int(1024),
                        required=False)

    parser.add_argument('-att', '-attributes',
                        help='Attributes dataset file',
                        default=configuration['embedding_attributes'],
                        required=False)

    parser.add_argument('-cnn', '-cnn_features',
                        help='Features in dataset file',
                        default=str(configuration['embedding_image']),
                        required=False)

    parser.add_argument('-tenc', '-text_encoder',
                        default=str('bow'),
                        help='Textual Encoder: "bow", "lstm"', required=False)

    parser.add_argument('-verbose', default=False,
                        help='Verbose debug mode', required=False)

    return vars(parser.parse_args())

def build_model(opt_model):
    clf = None
    # Load SVM + Random Forest Model
    if opt_model == 'rfr':
        clf = RandomForestRegressor(n_estimators=configuration['n_estimators'],
                                     n_jobs=configuration['n_jobs'],
                                     verbose=10)

    elif opt_model == 'svm':
        cl_ = SVC(C=10, verbose=10, cache_size=100, max_iter=configuration['max_iter'])
        clf = OneVsRestClassifier(cl_, n_jobs=configuration['n_jobs'])
    elif opt_model == 'linear-svm':
        cl_ = LinearSVC(C=10, verbose=10)
        clf = OneVsRestClassifier(cl_, n_jobs=configuration['n_jobs'])
    elif opt_model == 'linear':
        #clf = LinearRegression(n_jobs=6)
        cl_ = LinearRegression()
        clf = OneVsRestClassifier(cl_, n_jobs=configuration['n_jobs'])

    elif opt_model == 'logistic':
        #clf = LinearRegression(n_jobs=6)
        clf = LogisticRegression(C=10)

    return clf

def main(args):

    # file_estimation_classes = dir_data + 'attributes/predicate-matrix-binary.txt'
    print '#'*50, '\n'
    print 'Configuration: \n', json.dumps(configuration, sort_keys=True, indent=4)
    print '#'*50, '\n'
    print '-'*50, '\nLoading data ...\n', '-'*50
    print '\n', '#'*50

    # Load attributes
    attributes_data = np.loadtxt(configuration['embedding_attributes'])
    # Normalize [-1, +1]
    attributes_data = (attributes_data * 2) - 1
    print 'Attributes shape: ', attributes_data.shape

    #Load googlenet
    start = time.time()
    #cnn_data = np.loadtxt(cnn_dataset_file)
    cnn_data = np.array(h5py.File(configuration['embedding_image'],'r')['vgg'])
    cnn_data = np.ascontiguousarray(cnn_data)
    print 'CNN shape: ', cnn_data.shape
    print 'Time (load dataset): %f' % (time.time() - start)

    print '-'*50, '\nData loaded ...\n', '-'*50

    #Load class centroids estimations
    if configuration['dataset'] == './data/cub/':
        file_estimation_classes = configuration['dataset'] + 'attributes/class_attribute_labels_continuous.txt'
        estimation_attributes = np.loadtxt(file_estimation_classes)/100
    elif configuration['dataset'] == './data/awa/':
        file_estimation_classes = configuration['dataset'] + 'attributes/predicate-matrix-binary.txt'
        estimation_attributes = np.loadtxt(file_estimation_classes) / 100

    labels, id_labels, train_set, valid_set, test_set = collect_splits(configuration['dataset'] + 'features/')
    with open('./data/awa/features/tmp/ids_classname.json') as outp:
        id_class = json.load(outp)
    with open('./data/awa/features/tmp/predicates.json') as outp:
        predicates = json.load(outp)

    pred_labels = []
    pred_labels.append([predicates[str(i)] for i in range(85)])


    X_train = cnn_data[train_set]#[:1000]
    A_train = attributes_data[train_set]#[:1000]
    Y_train = id_labels[train_set]#[:1000]

    X_valid = cnn_data[valid_set]
    A_valid = attributes_data[valid_set]
    Y_valid = id_labels[valid_set]

    X_test = cnn_data[test_set]
    A_test = attributes_data[test_set]
    Y_test = id_labels[test_set]

    labels_test = np.unique(Y_test)

    del cnn_data

    # Embedding model
    print '-' * 50, '\nTraining embedding model\n', '-' * 50
    # clf = build_model(configuration['baseline_model'])
    # start = time.time()
    # clf.fit(X_train, A_train)

    clf = train_linear_svm(X_train, A_train)

    # print 'Acc (training): ', metrics.accuracy_score(A_train, embd_train)
    # print 'Acc (validation): ', metrics.accuracy_score(A_valid, embd_valid)
    # print 'Acc (test): ', metrics.accuracy_score(A_test, embd_test)
    #
    # knn = KNeighborsClassifier(n_neighbors=configuration['#neighbors'])
    # knn.fit(attributes_data, id_labels)
    #
    #
    # zsl_sanity = knn.predict(attributes_data)
    # zsl_train = knn.predict(embd_train)
    # zsl_valid = knn.predict(embd_valid)
    # zsl_test = knn.predict(embd_test)
    #
    # print 'Acc (sanity check):' , metrics.accuracy_score(id_labels, zsl_sanity)
    # print 'Acc (training): ', metrics.accuracy_score(Y_train, zsl_train)
    # print 'Acc (validation): ', metrics.accuracy_score(Y_valid, zsl_valid)
    # print 'Acc (test): ', metrics.accuracy_score(Y_test, zsl_test)
    #
    # fig = plt.figure(figsize=(30,5))
    # plt.axis([0, 85, 0, 1])
    # plt.bar(range(85), acc_test)
    # plt.plot(range(85), np.zeros(85) + acc_test.mean(), 'r')



    print 'Time (load training): %f' % (time.time() - start)

    print '-' * 50, '\nSaving embedding model\n', '-' * 50
    file_ = configuration['output_file'][:-3] + 'pkl'
    _ = joblib.dump(clf, file_, compress=9)



    """
    evaluation['coeficient_determination_train'] = clf.score(X_train,
                                                             A_train)
    evaluation['coeficient_determination_valid'] = clf.score(X_valid,
                                                             A_valid)
    evaluation['coeficient_determination_test'] = clf.score(X_test,
                                                            A_test)

    print 'Coeficient of Determination (training): ', evaluation['coeficient_determination_train']
    print 'Coeficient of Determination (validation): ', evaluation['coeficient_determination_valid']
    print 'Coeficient of Determination (testing): ', evaluation['coeficient_determination_test']
    """
    print '-' * 50, '\nEmbedding model Trained\n', '-' * 50

    # Embedding features
    # embd_test = clf.predict(X_test)
    # embd_train = clf.predict(X_train)
    # embd_valid = clf.predict(X_valid)

    # embd_train, acc_train = predict_svm(clf, X_train, A_train)
    # embd_valid, acc_valid = predict_svm(clf, X_valid, A_valid)
    embd_test, acc_test = predict_svm(clf, X_test, A_test)



    fig = plt.figure(figsize=(30,5))
    plt.axis([0, 85, 0, 1])
    plt.bar(range(85), acc_test)
    plt.plot(range(85), np.zeros(85) + acc_test.mean(), 'r')
    plt.xticks(range(85), pred_labels[0], rotation='vertical')
    # plt.savefig('./experiments/awa/%s_%s_test_acc.png'% (configuration['baseline_model'], configuration['tag'] ))

    # Distance model
    print '-' * 50, '\nTraining distance model\n', '-' * 50
    knn = KNeighborsClassifier(n_neighbors=configuration['#neighbors'])
    if configuration['estimated_values']:
        print '>> Training with estimated values'
        knn.fit(estimation_attributes, labels)
    else:
        print '>> Training with attributes data'
        #knn.fit(attributes_data, id_labels)
        knn.fit(A_test, Y_test)
    print '-' * 50, '\nDistance model trained\n', '-' * 50

    # Apply distance model
    print '-' * 50, '\nRunning Zero-Shot Learning\n', '-' * 50
    zsl_test = knn.predict(embd_test)
    #zsl_train = knn.predict(embd_train)
    #zsl_valid = knn.predict(embd_valid)

    print '-' * 50, '\nRunning Evaluation\n', '-' * 50
    evaluation['accuracy_test'] = metrics.accuracy_score(Y_test,
                                                         zsl_test)
    evaluation['precision_test'] = metrics.precision_score(Y_test,
                                                           zsl_test,
                                                           labels=labels,
                                                           average='weighted')
    evaluation['recall_test'] = metrics.recall_score(Y_test,
                                                     zsl_test,
                                                     labels=labels,
                                                     average='weighted')

    cm_ = metrics.confusion_matrix(Y_test,
                                   zsl_test,
                                   labels=labels_test)
    np.savetxt(configuration['output_file'] + '.confusion_test', cm_, fmt='%d')

    test_classes = []
    test_classes.append([str(id_class[str(ic+1)]) for ic in labels_test])
    fig = plt.figure(figsize=(30,30))
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm_)
    fig.colorbar(cax)
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xticklabels(test_classes[0], rotation='vertical')
    ax.set_yticklabels(test_classes[0], rotation='horizontal')
    fig.savefig('./experiments/awa/%s_%s_confusion_matrix.png'% (configuration['baseline_model'], configuration['tag'] ))


    print '-' * 50, '\nRunning Evaluation per Class\n', '-' * 50

    eval_per_class = {}
    matrix_results = {}
    for c in labels_test:

        y = (Y_test == c) * 1
        y_ = (zsl_test == c) * 1
        acc_ = metrics.accuracy_score(Y_test == c,
                                      zsl_test == c)

        pr_ = metrics.precision_score(Y_test == c,
                                      zsl_test == c)

        re_ = metrics.recall_score(Y_test == c,
                                   zsl_test == c)

        cm_ = metrics.confusion_matrix(Y_test == c,
                                       zsl_test == c)
        #roc from Lampert
        tpr_,fpr_, auc_ = roc(None, y_, y)

        eval_per_class[c] = { 'accuracy': acc_,
                              'precision': pr_,
                              'recall': re_
                              }
        matrix_results[c] = {'confusion_matrix': cm_,
                             'fpr': fpr_,
                             'tpr': tpr_,
                             'auc': auc_}


    evaluation['~evaluation_per_class'] = eval_per_class

    print json.dumps(evaluation, sort_keys=True, indent=4)

    with open(configuration['output_file'], 'w') as outfile:
        obj_ = {'evaluation':evaluation, 'configuration': configuration}
        json.dump(obj_, outfile, sort_keys=True, indent=4)

    # plot roc
    plot_roc(matrix_results, id_class)
    print matrix_results


    # plt.clf()
    # fig = plt.figure(figsize=(30,15))
    # plt.plot([0, 1], [0, 1], 'k--')
    #
    # keys = matrix_results.keys()
    # keys.sort()
    # for key in keys:
    #     tpr_ = matrix_results[key]['tpr']
    #     fpr_ = matrix_results[key]['fpr']
    #     auc_ = matrix_results[key]['auc']
    #
    #     plt.plot(fpr_, tpr_, label='%s (%d)[auc: %.2f]' % (id_class[str(key + 1)], key + 1, auc_))
    #
    # plt.xlabel('False positive rate')
    # plt.ylabel('True positive rate')
    # plt.title('ROC curve')
    # plt.legend(loc='best')
    # plt.show()
    # fig.savefig('./experiments/awa/%s_%s_roc_curve.png' % (configuration['baseline_model'], configuration['tag'] ))


    return evaluation, clf




    #model.fit(cnn_data[id_train_samples], attributes_data[id_train_samples])

    #attribute_predicted = model.predict(cnn_data[id_test_samples])
    # Load KNN Model
    #knn.predict(attribute_predicted, est_classes)


    # # Load text encoder
    # if(t_encoder == 'bow'):
    #     dir_txt_features = dir_data + 'features/bow_fine_grained/'
    #
    #     lst_files = file_utils.get_files(dir_txt_features, 'h5')
    #     t_data = h5py.File(dir_txt_features + lst_files[0], 'r')
    #     input_dim = t_data[lst_files[0][:-3]][t_data[lst_files[0][:-3]].keys()[0]].shape[1]
    #
    #     txt_encoder = bow.bow_encoder(input_dim, 1024)
    # dir_img_features = dir_data + 'images/'




    # Load image encoder
    # googlenet_model = googlenet.googlenet_encoder()
    # img_encoder = Model(input=googlenet_model.layers[0].input,
    #                                   output=googlenet_model.get_layer('flatten_3').output)

    # Generate Zero-Shot Learning model
    # dot_product = Merge(mode='dot', dot_axes=(1, 1))([txt_encoder.output, img_encoder.output])
    # zsl_model = Model(input=[txt_encoder.input, img_encoder.input], output=[dot_product])
    # zsl_model.compile(optimizer='rmsprop',
    #                 loss='categorical_crossentropy')


    # return zsl_model


if __name__ == '__main__':
    print '\n\n\nInitializing application...\n\n'

    random.seed(0)
    args = load_args()

    t_global = time.time()
    evaluation, clf = main(args)
    print 'Total (time): %f' % (time.time() - t_global)


    print '---\nClosing application ...\n'
