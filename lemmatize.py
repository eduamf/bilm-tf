#!/usr/bin/python3
from pandas import read_csv
import sys
from ufal.udpipe import Model, Pipeline
from unify import unify_sym

functional = set('ADP AUX CCONJ DET PART PRON SCONJ PUNCT NUM'.split())
print('Functional tags:', functional, file=sys.stderr)


def tag(text):
    try:
        processed = pipeline.process(text)
        output = [l.split('\t') for l in processed.split('\n') if not l.startswith('#')]
        output = [l for l in output if len(l) == 10]
        tagged = [w[2].lower() for w in output]
        #tagged = ['_'.join([w[2].lower(), w[3]]) for w in output]
        return ' '.join(tagged)
    except:
        pass


dataset = sys.argv[1]
modelfile = sys.argv[2]
model = Model.load(modelfile)
pipeline = Pipeline(model, 'tokenize', Pipeline.DEFAULT, Pipeline.DEFAULT, 'conllu')

df = read_csv(dataset, sep="\t", encoding="utf-8")

print('Tagging left context', file=sys.stderr)
df['tagged_left_context'] = df['left'].apply(lambda x: tag(x))
print('Tagging words', file=sys.stderr)
df['tagged_word'] = df['word'].apply(lambda x: tag(x))
print('Tagging right context', file=sys.stderr)
df['tagged_right_context'] = df['right'].apply(lambda x: tag(x))

print('Saving the results...', file=sys.stderr)
print(df.to_csv(sep="\t", encoding="utf-8"))
