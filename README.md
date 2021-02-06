# Orpheus
Music genre classification is a very complex and hard task even for humans. Main
obstacle is most commonly a very strong similarity between some genres like, for
example, Pop and R&B. Music genre classification models usually work with songs
audio but in our project we tried to use songs lyrics to classify the song into the right
genre.

It is to be expected that songs from the same genre will more or less have the same
themes. To give an example, country songs are usually about home and land. Metal
songs, on the other hand, mostly have darker themes like death and life.
This problem has been very well researched by many NLP experts. Currently best
results have been achieved by [Alexandros Tsaptsinos](https://ccrma.stanford.edu/groups/meri/assets/pdf/tsaptsinos2017preprint.pdf) who used a hierarchical
attention network which gave fantastic results.

## Data
In our project we used Metrolyrics dataset which contains more than 380 000 labeled
songs. This dataset consists of songs produced between 1970 and 2016. Each
dataset entry carries information about song and artist name, genre and lyrics.
Originally, this dataset also contained spanish songs and this would be very
problematic for our model but luckily, we found a [filtered out version](https://github.com/hiteshyalamanchili/SongGenreClassification/blob/master/dataset/english_cleaned_lyrics.zip) of the dataset.

Besides spanish songs, we also had to filter out songs without lyrics since they are
of no significance for us. We also had to “clean” lyrics so they don’t contain parts
which mark the end of the chorus and similar. Dataset can be downloaded [here](https://drive.google.com/file/d/1nPwSEiC4FnfA3tSe_ly3kkxB3XbgBZqY/view).

For word representation, we used pre-trained 100-dimensional GloVe
representations. They can be downloaded [here](http://nlp.stanford.edu/data/glove.6B.zip).

## Hierarchical Attention Network
As mentioned before, currently the best solution for the lyrics genre classification
problem is [Hierarchical Attention Network (HAN)](https://www.cs.cmu.edu/~./hovy/papers/16HLT-hierarchical-attention-networks.pdf). Because of that we decided to
implement such a model.

Main idea of the HAN model is simple: first read the context of each line individually
and then combine these contexts to get a “full” context of the song. But how exactly
does the HAN do that? It does it by using two layers of bidirectional gated recurrent
unit (GRU) followed by the attention. First layer is then used to “read” the
relationship between words in the line and the second one is used to get the full
context by combining all line contexts. In the end, we use a linear layer to do a
simple genre classification.

## Training
If you want to run the training yourself run the following:
```bash
python train.py [DATA PATH] --pretrain_file [PRETRAINED WORD VECTOR REPRESENTATION] --gpus 1 --benchmark --max_epochs [NUMBER OF TRAINING EPOCHS]
```
In case you don't use pretrained word representations, embeddings will be learned during the training but in that case expect worse results.

## Literature
* Lyrics-Based Music Genre Classification using a Hierarchical Attention Network; Alexandros Tsaptsinos; 2017.
* Hierarchical Attention Networks for Document Classification; Zichao Yang, Diyi Yang, Chris Dyer, Xiaodong He, Alex Smola, Eduard Hovy; 2016.
* Hierarchical attention networks for information extraction from cancer pathology reports; Shang Gao, Michael T Young, John X Qiu, Hong-Jun Yoon, James B Christian, Paul A Fearn, Georgia D Tourassi, Arvind Ramnthan; 2018.
