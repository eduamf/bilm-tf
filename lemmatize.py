#!/usr/bin/python3
from pandas import read_csv
import sys
from ufal.udpipe import Model, Pipeline

functional = set('ADP AUX CCONJ DET PART PRON SCONJ PUNCT NUM'.split())
print('Functional tags:', functional, file=sys.stderr)


def tag(text):
    processed = pipeline.process(text)
    output = [l.split('\t') for l in processed.split('\n') if not l.startswith('#')]
    output = [l for l in output if len(l) == 10]
    tagged = ['_'.join([w[2].lower(), w[3]]) for w in output]
    return ' '.join(tagged)


dataset = sys.argv[1]
modelfile = sys.argv[2]
model = Model.load(modelfile)
pipeline = Pipeline(model, 'tokenize', Pipeline.DEFAULT, Pipeline.DEFAULT, 'conllu')

df = read_csv(dataset, sep="\t", encoding="utf-8")

print('Tagging words', file=sys.stderr)
df['tagged_word'] = df['word'].apply(lambda x: tag(x))
print('Tagging contexts', file=sys.stderr)
df['tagged_context'] = df['context'].apply(lambda x: tag(x))

print('Saving the results...', file=sys.stderr)
print(df.to_csv(sep="\t", encoding="utf-8"))
