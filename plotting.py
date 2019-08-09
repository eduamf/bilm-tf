# python3
# coding: utf-8

import sys
from smart_open import open
import os
import numpy as np
import pylab as plot

if __name__ == '__main__':
    files2process = sys.argv[2:]
    lang = sys.argv[1]
    data = {}
    for f in files2process:
        name = os.path.basename(f).split('.')[0].replace('_target', '').\
            replace(lang.lower(), '').replace('_', ' ').strip()
        data[name] = []
        with open(f) as source:
            for l in source.readlines():
                if l:
                    if l.strip().startswith('Average F1 on 5-fold cross-validation:'):
                        f1 = l.strip().split(':')[1].split('(')[0].strip()
                        data[name].append(float(f1))

    plot.clf()
    vectors = []
    labels = []
    for entity in data:
        vector = data[entity]
        vectors.append(vector)
        labels.append(entity)
        print('Average F1 value for %s: %0.3f (+/- %0.3f)'
              % (entity, float(np.mean(vector)), float(np.std(vector))))

    plot.boxplot(vectors, labels=labels, whis='range')
    plot_title = 'WSD performance for ' + lang
    plot.title(plot_title)
    plot.xlabel('')
    plot.ylabel('Macro F1 scores')
    # plot.show()
    plot.savefig(lang + '.png', dpi=300)
    plot.close()
    plot.clf()
