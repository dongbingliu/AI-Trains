
""" 
@author:Jerry Liu
@file: EmotionClassification.py
@version:
@time: 2018/12/17 
@email:liudongbing8@163.com
@function： 情感分类
"""
import numpy as np
import time
import nltk
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import metrics
import Model
import jieba
from sklearn.model_selection import train_test_split

import WordToVec


positive_list = []
positive_label = []

negative_list = []
negative_label = []


def openFile(filePath, label=0):
    data_list = []
    label_list = []
    with open(filePath, encoding="utf-8", errors="ignore") as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip()
            # line = list(jieba.cut(line))
            line = nltk.word_tokenize(line)
            # print(line)
            data_list.append(line)
            label_list.append(label)

    return data_list, label_list


def main():
    positive_list, positive_label = openFile("datas/rt-polarity.pos", label=1)
    negative_list, negative_label = openFile("datas/rt-polarity.neg", label=0)

    all_list = positive_list + negative_list
    all_label = positive_label + negative_label

    all_list = np.asarray(all_list)
    all_label = np.asarray(all_label)

    print(type(all_list))
    word2Vec = WordToVec.WordToVector()
    x_datas = word2Vec.tfWord2Vector(all_list)
    y_datas = all_label

    # 混淆数据
    x_y_datas = np.column_stack([x_datas, y_datas])
    np.random.shuffle(x_y_datas)

    x_datas = x_y_datas[:, :-1]
    y_datas = x_y_datas[:, -1]

    x_train, x_test, y_train, y_test = train_test_split(x_datas, y_datas, test_size=0.2)

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

    for classifier in test_classifiers:
        print('******************* %s ********************' % classifier)
        start_time = time.time()
        model = classifiers[classifier](x_train, y_train)

        print('training took %fs!' % (time.time() - start_time))
        predict = model.predict(x_test)
        # if model_save_file != None:
        #     model_save[classifier] = model
        # if is_binary_class:
        #     precision = metrics.precision_score(test_y, predict)
        #     recall = metrics.recall_score(test_y, predict)
        #     print('precision: %.2f%%, recall: %.2f%%' % (100 * precision, 100 * recall))
        accuracy = metrics.accuracy_score(y_test, predict)
        print('accuracy: %.2f%%' % (100 * accuracy))

        # print("Start Training......")
        # startTime = time.time()
        # model = Model.decision_tree_classifier(x_train, y_train)
        # print("EndTraining......")
        # endTime = time.time()
        # print("Train cost Time = ", endTime - startTime)
        # # score = model.score(x_test, y_test)
        # print("tf-id去除停用词特征_NB_:\n", classification_report(y_test, model.predict(x_test)))
        #
        # import utils.confusion_matrix_png_util
        #
        # print("混淆矩阵表示：\n",confusion_matrix(y_test, model.predict(x_test)))


        # print("[score] = ", score)


if __name__ == "__main__":
    main()
