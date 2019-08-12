# python3
# coding: utf-8

import argparse
import warnings
from collections import Counter
from sklearn.metrics import classification_report, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_validate
from wsd_helpers import *
import random

warnings.filterwarnings("ignore")


def classify(data_file, w2v=None, elmo=None, max_batch_size=30, algo='logreg'):
    data, mfs_dic = load_dataset(data_file)
    scores = []
    f_scores = []
    random_f_scores = []

    # data looks like {w1 = [[w1 context1, w1 context2, ...], [w2 context1, w2 context2, ...]], ...}
    for word in data:
        x_train = []
        y = []
        if elmo:
            batcher, sentence_character_ids, elmo_sentence_input = elmo
            sentences = [tokenize(el[0]) for el in data[word]]
            nums = [el[1] for el in data[word]]
            y = [el[2] for el in data[word]]
            input_data = [(s, n) for s, n in zip(sentences, nums)]
            print('=====')
            print('%s: %d sentences total' % (word, len(sentences)))
            print('=====')
            # Here we divide all the sentences for the current word in several chunks
            # to to reduce the batch size
            with tf.Session() as sess:
                # It is necessary to initialize variables once before running inference.
                sess.run(tf.global_variables_initializer())
                for chunk in divide_chunks(input_data, max_batch_size):
                    chunk_sentences = [el[0] for el in chunk]
                    chunk_nums = [el[1] for el in chunk]
                    x_train += get_elmo_vector(
                        sess, chunk_sentences, batcher, sentence_character_ids,
                        elmo_sentence_input, chunk_nums)
        else:
            print('=====')
            print('%s' % word)
            print('=====')
            tp = 0
            mfs = int(mfs_dic[word])
            print('MFS', mfs)
            examples = len(data[word])
            for instance in data[word]:
                sent, num, cl = instance
                if cl == mfs:
                    tp += 1
                if w2v:
                    vect = get_word_vector(tokenize(sent), w2v, num)
                else:
                    vect = get_dummy_vector()
                x_train.append(vect)
                y.append(cl)
            f = f1_score(y, [int(mfs_dic[word])]*examples, average='macro')
            f_scores.append(f)
            print('F1 score is ', f)
            print('TP and all examples', tp, examples)
            all_senses = list(set(y))
            f_random = f1_score(y, [random.choice(all_senses) for ex in range(examples)], average='macro')
            print('Random F1 score is ', f_random)
            random_f_scores.append(f_random)


        classes = Counter(y)
        print('Distribution of classes in the whole sample:', dict(classes))

        if algo == 'logreg':
            clf = LogisticRegression(
                solver='lbfgs', max_iter=1000, multi_class='auto', class_weight='balanced')
        else:
            clf = MLPClassifier(hidden_layer_sizes=(200, ), max_iter=500)
        averaging = True  # Do you want to average the cross-validate metrics?

        scoring = ['precision_macro', 'recall_macro', 'f1_macro']
        # some splits are containing samples of one class, so we split until the split is OK
        counter = 0
        while True:
            try:
                cv_scores = cross_validate(clf, x_train, y, cv=5, scoring=scoring)
            except ValueError:
                counter += 1
                if counter > 500:
                    print('Impossible to find a good split!')
                    exit()
                continue
            else:
                # No error; stop the loop
                break
        scores.append([cv_scores['test_precision_macro'].mean(),
                       cv_scores['test_recall_macro'].mean(), cv_scores['test_f1_macro'].mean()])
        if averaging:
            print("Average Precision on 5-fold cross-validation: %0.3f (+/- %0.3f)" % (
                cv_scores['test_precision_macro'].mean(),
                cv_scores['test_precision_macro'].std() * 2))
            print("Average Recall on 5-fold cross-validation: %0.3f (+/- %0.3f)" % (
                cv_scores['test_recall_macro'].mean(),
                cv_scores['test_recall_macro'].std() * 2))
            print("Average F1 on 5-fold cross-validation: %0.3f (+/- %0.3f)" % (
                cv_scores['test_f1_macro'].mean(),
                cv_scores['test_f1_macro'].std() * 2))
        else:
            print("Precision values on 5-fold cross-validation:", file=sys.stderr)
            print(cv_scores['test_precision_macro'], file=sys.stderr)
            print("Recall values on 5-fold cross-validation:", file=sys.stderr)
            print(cv_scores['test_recall_macro'], file=sys.stderr)
            print("F1 values on 5-fold cross-validation:", file=sys.stderr)
            print(cv_scores['test_f1_macro'], file=sys.stderr)

        print('\n')

    print('=====')
    print('Average precision value for all words: %0.3f (+/- %0.3f)' %
          (float(np.mean([x[0] for x in scores])), np.std([x[0] for x in scores]) * 2))
    print('Average recall value for all words: %0.3f (+/- %0.3f)' %
          (float(np.mean([x[1] for x in scores])), np.std([x[1] for x in scores]) * 2))
    print('Average F1 value for all words: %0.3f (+/- %0.3f)' %
          (float(np.mean([x[2] for x in scores])), np.std([x[2] for x in scores]) * 2))
    print('Average F1 value for all words with MFS: %0.3f (+/- %0.3f)' %
          (float(np.mean(f_scores)), np.std(f_scores) * 2))
    print('Average random F1 value for all words: %0.3f (+/- %0.3f)' %
            (float(np.mean(random_f_scores)), np.std(random_f_scores) * 2))

    return scores


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--input', help='Path to tab-separated file with WSD data', required=True)
    arg('--w2v', help='Path to word2vec model (optional)')
    arg('--elmo', help='Path to ELMo model (optional)')
    parser.set_defaults(w2v=False)
    parser.set_defaults(elmo=False)

    args = parser.parse_args()
    data_path = args.input

    if args.w2v:
        emb_model = load_word2vec_embeddings(args.w2v)
        eval_scores = classify(data_path, w2v=emb_model)
    elif args.elmo:
        emb_model = load_elmo_embeddings(args.elmo)
        eval_scores = classify(data_path, elmo=emb_model)
    else:
        eval_scores = classify(data_path)
