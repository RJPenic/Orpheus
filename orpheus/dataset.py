import torch
from dataclasses import dataclass, field
from typing import List
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import csv

@dataclass
class Instance:
	label : int
	text : List[List[str]] = field(default_factory = list)

	def __init__(self, label : int, text : List[List[str]]):
		self.label = label
		self.text = text

class LyricsDataset(torch.utils.data.Dataset):
	def __init__(self, instances, max_vocab_size = 30000, max_lines = 30, max_words_per_line = 10, remove_stop_words = False):
		self.instances = instances

		ct_txt = {}

		for instance in instances:
			for line in instance.text:
				for token in line:
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

	def encode(self, instance_lyrics):
		encoded = []
		pad_lines = True
		for j, line in enumerate(instance_lyrics):
			if j >= self.max_lines:
				pad_lines = False
				break

			temp = []
			pad_words = True
			for i, token in enumerate(line):
				if i >= self.max_words_per_line:
					pad_words = False
					break

				temp.append(self.stoi.get(token, self.stoi["<UNK>"]))

			if pad_words:
				temp.extend([self.stoi["<PAD>"]] * (self.max_words_per_line - i - 1)) # maybe put pad in front ???
			
			encoded.append(temp)

		if pad_lines:
				encoded.extend([[self.stoi["<PAD>"]] * self.max_words_per_line] * (self.max_lines - j - 1))

		return torch.Tensor(encoded).type(torch.LongTensor)


# ------ ˇˇˇ IGNORE ˇˇˇ ------
if __name__ == "__main__":
	filepath = '../dataset/lyrics_1.csv'

	ds = LyricsDataset.from_file(filepath, ["Rock", "Pop", "Hip-Hop", "Metal", "Country", "Jazz", "Electronic", "R&B", "Indie", "Folk", "Other"])
	print(ds[0])