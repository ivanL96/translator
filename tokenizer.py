import numpy as np
import os

# def FreqDict2List(dt):
	# return sorted(dt.items(), key=lambda d:d[-1], reverse=True)
 
class TokenList:
    def __init__(self, token_list):
        self.id2t = ['<PAD>', '<UNK>', '<S>', '</S>'] + token_list
        self.num = len(self.id2t)
        self.t2id = {v:k for k,v in enumerate(self.id2t)}
        self.startid = 2
        self.endid = 3

    def id(self, x):
        return self.t2id.get(x, 1)
  
    def token(self, x):
        return self.id2t[x]
	
def pad_to_longest(sequence, tokens, max_len=999):
	longest = min( len(max(sequence, key=len)) + 2, max_len )
	X = np.zeros((len(sequence), longest), dtype='int32')
	X[:,0] = tokens.startid
	for i, x in enumerate(sequence):
		x = x[:max_len-2]
		for j, z in enumerate(x):
			X[i,1+j] = tokens.id(z)
		X[i,1+len(x)] = tokens.endid
	return X

def makeS2Sdict(data, min_freq=5, delimiter=' ', dict_file=None):
    if dict_file and os.path.exists(dict_file):
        print('loading', dict_file)
        with open(dict_file, encoding="utf-8") as fin:
            lst = list(ll for ll in fin.read().split('\n') if ll != "")
        midpos = lst.index('<@@@>')
        itokens = TokenList(lst[:midpos])
        otokens = TokenList(lst[midpos+1:])
        return itokens, otokens

    wdicts = [{}, {}]
    for ss in data:
        for seq, wd in zip(ss, wdicts):
            for w in seq.split(delimiter): 
                wd[w] = wd.get(w, 0) + 1

    wlists = []
    for wd in wdicts:	
        # wd = FreqDict2List(wd)
        wd = sorted(wd.items(), key=lambda d:d[-1], reverse=True)
        wlist = [x for x,y in wd if y >= min_freq]
        wlists.append(wlist)
    print('seq 1 words:', len(wlists[0]))
    print('seq 2 words:', len(wlists[1]))
    itokens = TokenList(wlists[0])
    otokens = TokenList(wlists[1])
    if dict_file is not None:
        with open(dict_file, "w", encoding = "utf-8") as fout:
            for k in wlists[0]+['<@@@>']+wlists[1]:
                fout.write(str(k) + "\n")
    return itokens, otokens

def MakeS2SData(data, itokens=None, otokens=None, delimiter=' ', max_len=200):
    """ splits data element in two sequences """
    Xs = [[], []]
    for ss in data:
        for seq, xs in zip(ss, Xs):
            xs.append(list(seq.split(delimiter)))
    X, Y = pad_to_longest(Xs[0], itokens, max_len), pad_to_longest(Xs[1], otokens, max_len)
    return X, Y

# def S2SDataGenerator(fn, itokens, otokens, batch_size=64, delimiter=' ', max_len=999):
# 	Xs = [[], []]
# 	while True:
# 		for ss in LoadCSV(fn, gen=True):
# 			for seq, xs in zip(ss, Xs):
# 				xs.append(list(seq.split(delimiter)))
# 			if len(Xs[0]) >= batch_size:
# 				X, Y = pad_to_longest(Xs[0], itokens, max_len), pad_to_longest(Xs[1], otokens, max_len)
# 				yield [X, Y], None
# 				Xs = [[], []]