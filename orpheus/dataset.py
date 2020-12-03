import torch
from dataclasses import dataclass, field
from typing import List
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import csv

from nltk.corpus import stopwords

@dataclass
class Instance:
    label : int
    text : List[List[str]] = field(default_factory = list)

    def __init__(self, label : int, text : List[List[str]]):
        self.label = label
        self.text = text

class LyricsDataset(torch.utils.data.Dataset):
    def __init__(self, instances, max_vocab_size = 33000, max_lines = 30, max_words_per_line = 10, remove_stop_words = False):
        self.instances = instances
        self.stop_words = stopwords.words('english')
        self.stopwords.update(['you\'re', 'i\'m', 'she\'s', 'he\'s', 'it\'s'])

        ct_txt = {}

        for instance in instances:
            for line in instance.text:
                for token in line:
                    if remove_stop_words and token in self.stopwords:
                        ct_txt[token] = ct_txt.get(token, 0) + 1

        self.text_vocab = Vocab(ct_txt, max_lines, max_words_per_line, max_vocab_size)
        
    def from_file(filename, labels, max_lines = 30, max_words_per_line = 10, skip_first_line = True, remove_stop_words = False):
        instances = []
        labels = [label.lower() for label in labels]

        with open(filename) as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)

            for i, row in enumerate(csv_reader):
                if i == 0 and skip_first_line:
                    continue

                instances.append(Instance(
                    int(labels.index(row[5].lower())),
                    [line.split() for line in row[6].split('\n')]
                ))

        return LyricsDataset(instances, len(labels), max_lines, max_words_per_line, remove_stop_words)

    def __getitem__(self, i):
        return self.text_vocab.encode(self.instances[i].text), self.instances[i].label
    
    def __len__(self):
        return len(self.instances)

class Vocab:
    def __init__(self, frequencies, max_lines, max_words_per_line, max_size = -1, min_freq = 0,
                special = ["<PAD>", "<UNK>"]): # maybe add additional special for line padding ???
        self.stoi = {}

        self.max_lines = max_lines
        self.max_words_per_line = max_words_per_line

        for s in special:
            self.stoi[s] = len(self.stoi)

        sorted_tokens = sorted(frequencies.keys(), key = lambda k: -frequencies[k])

        for t in sorted_tokens:
            if min_freq <= frequencies[t] or (len(self.stoi) < max_size or max_size <= 0):
                self.stoi[t] = len(self.stoi)

    def encode(self, text):
        encoded = []

        for j, line in enumerate(text):
            if j >= self.max_lines:
                break

            temp = []
            for i, token in enumerate(line):
                if i >= self.max_words_per_line:
                    break

                temp.append(self.stoi.get(token, self.stoi["<UNK>"]))
            
            encoded.append(temp)

        return encoded

def load_vec_file_to_dict(filename):
    with open(filename) as f:
        content = f.readlines()
        
    content = [x.strip() for x in content]
    
    vecs = {}
    
    for line in content:
        elems = line.split()
        vecs[elems[0]] = torch.Tensor([float(n) for n in elems[1:]])
        
    return vecs
        
    
def load_vec_repr(vocab, d = 300, file = None, freeze = False):
    emb_mat = torch.randn(len(vocab.stoi), d)
    emb_mat[0] = torch.zeros(d)

    if file is not None:
        vecs = load_vec_file_to_dict(file)
        
        for k in vocab.stoi:
            if k in vecs:
                emb_mat[vocab.stoi[k]] = vecs[k]


    return nn.Embedding.from_pretrained(emb_mat, padding_idx = 0, freeze = freeze)

def pad_collate_fn(batch, pad_index = 0): # ???
    texts, labels = list(zip(*batch))
    bsz = len(labels)

    nums_lines = [len(lines) for lines in texts]
    nums_words = [[len(line) for line in lines] for lines in texts]

    max_lines = max(nums_lines)
    max_words = max([max(nw) for nw in nums_words])

    texts_tensor = torch.full((bsz, max_lines, max_words), pad_index).long()
    line_lens_tensor = torch.full((bsz, max_lines), pad_index).long()

    for i, text in enumerate(texts):
        text_len = nums_lines[i]
        line_lens_tensor[i, :text_len] = torch.LongTensor(nums_words[i])
        for j, line in enumerate(text):
            line_len = nums_words[i][j]
            texts_tensor[i, j, :line_len] = torch.LongTensor(line)

    return texts_tensor, torch.LongTensor(labels), torch.LongTensor(nums_lines), line_lens_tensor

# ------ ˇˇˇ IGNORE ˇˇˇ ------
if __name__ == "__main__":
    filepath = '../dataset/lyrics_1.csv'

    ds = LyricsDataset.from_file(filepath, ["Rock", "Pop", "Hip-Hop", "Metal", "Country", "Jazz", "Electronic", "R&B", "Indie", "Folk", "Other"])
    print(ds[0])

    train_dataloader = DataLoader(dataset=ds, batch_size=2, 
                                  shuffle=False, collate_fn=pad_collate_fn)

    for text, labels, len1, len2 in train_dataloader:
        print(text)
        print(labels)
        print(len1)
        print(len2)
        break