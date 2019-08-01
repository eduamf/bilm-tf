import re
from bs4 import BeautifulSoup
from sklearn.preprocessing import LabelEncoder
import csv


path_test = 'EnglishLS.test'
path_train = 'EnglishLS.train'

def parse_englishls(path):
    all_data = []
    with open(path) as f_in:
        data = f_in.read()
        soup = BeautifulSoup(data, 'lxml')
        words = soup.find_all("lexelt")
    
        for word in words:
            if not word['item'].endswith('.n'):
                continue
            classes = []
            word_data = []
            for inst in word.children:
                if inst.name == 'instance':
                    for child in inst.children:
                        if child.name == 'answer':
                            sense_id = child['senseid']
                            classes.append(sense_id)
                        if child.name == 'context':
                            cont = child.get_text().strip()
                    word_data.append([word['item'], sense_id, cont])
            le = LabelEncoder()
            le.fit(classes)
            word_data = [[x[0], le.transform([x[1]])[0], x[2]] for x in word_data]
            all_data.append(word_data)
    return all_data
        

if __name__ == '__main__':
    all_data = parse_englishls(path_train)
    #data_test = parse_englishls(path_test)
    #all_data = data_train + data_test
    with open('english_ls.tsv', 'w') as f_out:
        ds = csv.writer(f_out, delimiter='\t')
        header = ['id', 'lemma', 'sense_id', 'left', 'word', 'right', 'senses']
        ds.writerow(header)
        i = 1
        for word in all_data:
            for instance in word:
                word, sense_id, context = instance
                lemma = word.split('.')[0]
                left_pos = context.lower().find(lemma)
                left = context[:left_pos]
                right = context[left_pos + len(lemma):]
                row = [str(i), lemma, str(sense_id),  left, lemma, right, str(None)]
                ds.writerow(row)
                i += 1
    
