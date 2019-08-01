# python3
# coding: utf-8

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
import csv
import numpy as np
import sys
import warnings
from collections import Counter

warnings.filterwarnings("ignore")


def classify(data_file):
    data = load_dataset(data_file)
    scores = []
    # data looks like {w1 = [[w1 context1, w1 context2, ...], [w2 context1, w2 context2, ...]], ...}
    for word in data:
        print(word)
        x_train = []
        y = []
        for instance in data[word]:
            sent, num, cl = instance
            vect = get_dummy_vector(sent, num)
            x_train.append(vect)
            y.append(cl)
        classes = Counter(y)
        print('Distribution of classes in the whole sample:', dict(classes))

        clf = LogisticRegression(solver='lbfgs', max_iter=1000, multi_class='multinomial')
        averaging = True  # Do you want to average the cross-validate metrics?

        scoring = ['precision_macro', 'recall_macro', 'f1_macro']
        # some splits are containing samples of one class, so we split until the split is OK
        while True:
            try:
                cv_scores = cross_validate(clf, x_train, y, cv=5, scoring=scoring)
            except ValueError:
                continue
            else:
                # No error; stop the loop
                break

        scores.append([cv_scores['test_precision_macro'].mean(),
                       cv_scores['test_recall_macro'].mean(), cv_scores['test_f1_macro'].mean()])
        if averaging:
            print("Average Precision on 5-fold cross-validation: %0.3f (+/- %0.3f)" % (
                cv_scores['test_precision_macro'].mean(),
                cv_scores['test_precision_macro'].std() * 2), file=sys.stderr)
            print("Average Recall on 5-fold cross-validation: %0.3f (+/- %0.3f)" % (
                cv_scores['test_recall_macro'].mean(),
                cv_scores['test_recall_macro'].std() * 2), file=sys.stderr)
            print("Average F1 on 5-fold cross-validation: %0.3f (+/- %0.3f)" % (
                cv_scores['test_f1_macro'].mean(),
                cv_scores['test_f1_macro'].std() * 2), file=sys.stderr)
        else:
            print("Precision values on 5-fold cross-validation:", file=sys.stderr)
            print(cv_scores['test_precision_macro'], file=sys.stderr)
            print("Recall values on 5-fold cross-validation:", file=sys.stderr)
            print(cv_scores['test_recall_macro'], file=sys.stderr)
            print("F1 values on 5-fold cross-validation:", file=sys.stderr)
            print(cv_scores['test_f1_macro'], file=sys.stderr)

        print('\n')

    print('Average precision value for all words: %0.3f (+/- %0.3f)' %
          (float(np.mean([x[0] for x in scores])), np.std([x[0] for x in scores]) * 2))
    print('Average recall value for all words: %0.3f (+/- %0.3f)' %
          (float(np.mean([x[1] for x in scores])), np.std([x[1] for x in scores]) * 2))
    print('Average F1 value for all words: %0.3f (+/- %0.3f)' %
          (float(np.mean([x[2] for x in scores])), np.std([x[2] for x in scores]) * 2))


def load_dataset(data_file):
    data = csv.reader(open(data_file), delimiter='\t')
    _ = next(data)
    data_set = {}
    cur_lemma = None
    word_set = []
    for row in data:
        i, lemma, sense_id, left, word, right, senses = row
        if lemma != cur_lemma:
            cur_lemma = lemma
            if len(word_set) > 0:
                data_set[cur_lemma] = word_set
            word_set = []
        sent = ' '.join([left, word, right])
        cl = int(sense_id)
        num = len(left.split(' '))
        word_set.append((sent, num, cl))
    return data_set


def get_dummy_vector(sent, num):
    vect = np.random.rand(1, 1024)
    return vect[0]


if __name__ == '__main__':
    data_path = sys.argv[1]  # tab-separated file
    classify(data_path)
