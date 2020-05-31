import random
import re


ru = 'rus.txt'
# de = 'en2de.s2s.valid.txt'
# TEST
NB_SAMPLES = 70000
NB_VALID_DATA = NB_SAMPLES // 10
NB_TEST = 40000

def clean(text_seq):
    clean_ss = []
    for text in text_seq:
        text = text.lower().replace('ё', 'е')
        text = re.compile(r'([^\s\w]|_)+').sub(' ', text)
        text = ' '.join(text.split())
        text = text[:-1] + ' .' if text[-1]==' ' else text
        clean_ss.append(text)
    return clean_ss

DATA = []
last_chars = []
with open(ru, encoding='utf-8') as fin:
    for line in fin.readlines():
        lln = line.replace('\n', '').split('\t')
        
        last_chars.append(lln[0][-1]) 
        last_chars.append(lln[1][-1])
        
        DATA.append(clean(lln))
    print('totallen', len(DATA))

last_chars = set(last_chars)
print(last_chars)
print([c for c in last_chars if not c.isalnum()])

print('data5', DATA[:5])
random.shuffle(DATA)
print('data5shuffled',DATA[:5])


VALID_DATA = DATA[:NB_VALID_DATA]
TRAIN_DATA = DATA[NB_VALID_DATA : NB_SAMPLES+NB_VALID_DATA]
TEST_DATA = DATA[NB_SAMPLES+NB_VALID_DATA : NB_SAMPLES+NB_VALID_DATA+NB_TEST]

# with open('valid.txt', 'w', encoding='utf-8') as f:
#     f.write('\n'.join(['\t'.join(line) for line in VALID_DATA]))
print(len(VALID_DATA), VALID_DATA[:5])
print(len(TRAIN_DATA), TRAIN_DATA[:5])
print(len(TEST_DATA), TEST_DATA[:5])
