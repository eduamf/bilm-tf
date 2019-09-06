import re
from sklearn.preprocessing import LabelEncoder
import csv
from xml.dom import minidom

path = 'senseval.xml'
    
def parse_english(path):
    doc = minidom.parse(path)
    instances = doc.getElementsByTagName("instance")
    classes = {}
    data = {}
    for i in instances:
        lex = i.getAttribute("id")
        answer = i.getElementsByTagName("answer")[0]
        sense = answer.getAttribute("senseid")
        if '.n' not in lex:
            continue
        lex = lex.split('.')[0]
        if lex in classes:
            classes[lex].append(sense)
        else:
            classes[lex] = [sense]
        contexts = i.getElementsByTagName("context") 
        for c in contexts:
            left = c.childNodes[0].data
            h = c.getElementsByTagName("head")
            for el in h: 
                target = el.childNodes[-1].data
            right = c.childNodes[-1].data
            if lex in data:
                data[lex].append([sense, left.strip(), target, right.strip()])
            else:
                data[lex] = [[sense, left.strip(), target, right.strip()]]
    cleaned_data = []
    for word in sorted(data):
        le = LabelEncoder()
        le.fit(classes[word])
        for context in data[word]:
            cleaned_data.append([word, le.transform([context[0]])[0], context[1], context[2], context[3]])
    return cleaned_data
        

if __name__ == '__main__':
    all_data = parse_english(path)
    with open('english.csv', 'w') as f_out:
        ds = csv.writer(f_out, delimiter='\t')
        header = ['id', 'lemma', 'sense_id', 'left', 'word', 'right', 'senses']
        ds.writerow(header)
        i = 1
        for instance in all_data:
            word, sense_id, left, head, right = instance
            row = [str(i), word, str(sense_id),  left, head, right, str(None)]
            ds.writerow(row)
            i += 1
    
