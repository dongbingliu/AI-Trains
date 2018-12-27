#!usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import time
from sklearn import metrics
import numpy as np
import pickle as pickle
import MachineLearning.DatasUtils as DatasUtils
import MachineLearning.Model as Model

if __name__ == '__main__':
    print('start...')
    data_file = "mnist.pkl.gz"
    thresh = 0.5
    model_save_file = None
    model_save = {}

    test_classifiers = ['NB', 'RF', 'DT', 'SVM', 'GBDT', 'LR', "KNN"]
    classifiers = {'NB': Model.naive_bayes_classifier,
                   'RF': Model.random_forest_classifier,
                   'DT': Model.decision_tree_classifier,
                   'SVM': Model.svm_classifier,
                   'SVMCV': Model.svm_cross_validation,
                   'GBDT': Model.gradient_boosting_classifier,
                   'LR': Model.logistic_regression_classifier,
                   'KNN': Model.knn_classifier
                   }

    print('reading training and testing data...')

    # 食物分类数据接口
    # train_x, train_y, test_x, test_y = DatasUtils.getFoodDatas()
    # 花卉分类数据接口
    # train_x, train_y, test_x, test_y = DatasUtils.getFlowerDatas()
    # 猫狗分类数据接口
    # train_x, train_y, test_x, test_y = DatasUtils.getAnimalsDatas()
    # 大规模物体分类数据接口
    train_x, train_y, test_x, test_y = DatasUtils.getMulObjectDatas()

    num_train, num_feat = train_x.shape
    num_test, num_feat = test_x.shape
    is_binary_class = (len(np.unique(train_y)) == 2)
    print('******************** Data Info *********************')
    print('#training data: %d, #testing_data: %d, dimension: %d' % (num_train, num_test, num_feat))

    for classifier in test_classifiers:
        print('******************* %s ********************' % classifier)
        start_time = time.time()
        model = classifiers[classifier](train_x, train_y)
        print('training took %fs!' % (time.time() - start_time))
        predict = model.predict(test_x)
        if model_save_file != None:
            model_save[classifier] = model
        if is_binary_class:
            precision = metrics.precision_score(test_y, predict)
            recall = metrics.recall_score(test_y, predict)
            print('precision: %.2f%%, recall: %.2f%%' % (100 * precision, 100 * recall))
        accuracy = metrics.accuracy_score(test_y, predict)
        print('accuracy: %.2f%%' % (100 * accuracy))

    if model_save_file != None:
        pickle.dump(model_save, open(model_save_file, 'wb'))
