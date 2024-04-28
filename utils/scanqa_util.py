class Answer(object):
    def __init__(self, answers=None, unk_token='<unk>', ignore_idx=-100):
        if answers is None:
            answers = []
        self.unk_token = unk_token
        self.ignore_idx = ignore_idx
        self.vocab = {x: i for i, x in enumerate(answers)}
        self.rev_vocab = dict((v, k) for k, v in self.vocab.items())

    def itos(self, i):
        if i == self.ignore_idx:
            return self.unk_token
        return self.rev_vocab[i]

    def stoi(self, v):
        if v not in self.vocab:
            #return self.vocab[self.unk_token]
            return self.ignore_idx
        return self.vocab[v]

    def __len__(self):
        return len(self.vocab) 