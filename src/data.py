import numpy as np
import operator
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from utils import *
import os
import pickle
import random
import csv
import re
from config import *
from tqdm import tqdm

csv.field_size_limit(2147483647)

class Data:
  def __init__(self):
    self._train_file_path = os.path.join('train.csv')
    self._test_file_path = os.path.join('test.csv')

    self.examples = []
    self.test = []

    with open(self._train_file_path) as inputfile, open(self._test_file_path) as testfile:
      reader = csv.reader(inputfile)
      records = [record for record in reader]
      for record in records[1:]:
        tid = record[0]
        text = record[1]
        author = record[2]
        self.examples.append((tid, text, author))

      reader = csv.reader(testfile)
      records = [record for record in reader]
      for record in records[1:]:
        tid = record[0]
        text = record[1]
        self.test.append((tid, text))

    print 'initial shuffling of examples'

    random.shuffle(self.examples)

    print 'intializing embeddings'

    self.embeddings = Embeddings(embedding_file_path, embedding_dim, vocab_size, self)
    mws = np.asarray([1,0,0], dtype='int32')
    eap = np.asarray([0,1,0], dtype='int32')
    hpl = np.asarray([0,0,1], dtype='int32')
    self.labels = {'MWS': mws, 'EAP': eap, 'HPL': hpl}
    len_examples = len(self.examples)
    print 'splitting validation and training examples'

    self.seq_length = seq_length
    self.batch_size = batch_size

    self.train_text_data = []
    self.train_targets = []
    self.train_pos_data = []

    for idx in tqdm(range(len_examples)):
      ex = self.examples[idx]
      i = get_embedding_input(ex[1], self.seq_length, (self.seq_length,), self.embeddings)
      pos = tags_input(ex[1], self.seq_length)
      self.train_text_data.append(i)
      self.train_pos_data.append(pos)
      self.train_targets.append(self.labels[ex[2]])

    print 'number of training examples ' + str(len(self.examples))
    
  def _input_generator(self, train=True):
    def iter():
      if not train:
        return self.val_examples
      else:
        random.shuffle(self.examples)
        return self.examples 

    ipt = iter()
    curr = 0
    reset = False
    while True:
      I = []
      targets = []

      if reset:
        ipt = iter()
        curr = 0
        reset = False

      for i in range(self.batch_size):
        try:
          if train and curr >= self.train_count:
            reset = True
            break
          elif not train and curr >= self.val_count:
            reset = True
            break
          text, label = ipt[curr][1], ipt[curr][2]
          curr += 1
        except StopIteration:
          reset = True
          break
        input_text = get_embedding_input(text, self.seq_length, (self.seq_length,), self.embeddings)
        target = self.labels[label]

        I.append(input_text)
        targets.append(target)

      if reset:
        continue
      
      yield ([np.asarray(I, dtype='int32'), np.asarray(I, dtype='int32'), np.asarray(I, dtype='int32')],
              np.asarray(targets, dtype='int32'))
      #yield (np.asarray(I, dtype='float32'), np.asarray(targets, dtype='int32'))
    yield None

  def input_generator(self, train=True):
    return self._input_generator(train)

class Embeddings:
  def __init__(self, embedding_file_path, dim, vocab_size, input_data):
    self.dim = dim
    self.vocab_size = vocab_size
    self.construct_embedding_matrix(input_data, embedding_file_path)

  def construct_embedding_matrix(self, input_data, embedding_file_path):
    embedding_file = open(embedding_file_path, 'r')
    self.embedding_index = {}

    print 'reading embedding file'

    for line in embedding_file:
      tokens = line.split()
      word = tokens[0]
      vector = np.asarray(tokens[1:], dtype='float32')
      self.embedding_index[word] = vector

    print 'tokenizing text'

    embedding_file.close()

    print "constructing the word index"
    self.construct_word_index(self.vocab_size, input_data)
    self.id_to_word = {}
    self.embedding_matrix = np.zeros((self.vocab_size+2, self.dim))
    print 'done creating word index'

    for word, i in self.word_index.items():
      self.id_to_word[i] = word
      embedding_vector = self.embedding_index.get(word)
      if embedding_vector is not None:
        self.embedding_matrix[i] = embedding_vector[:self.dim]

    print 'done creating embeddings'

  def construct_word_index(self, vocab_size, input_data):
    self.word_index = {}
    word_count = {}

    if word_index_file.strip(" ") != "":
      f = open('word_index.pkl')
      self.word_index = pickle.load(f)
      f.close()
      return 
    max_id = 0

    words = set() 
    d = []
    d.extend(input_data.examples)
    d.extend(input_data.test)
    for example in d:
      text = example[1]
      text = re.sub(r"([^\w\s])", r" \1 ", text)
      tokens = text_to_word_sequence(text, lower=True)
      for token in tokens:
        words.add(token)

    print '*&*&*&*&*&*&', len(words)
    for idx,word in enumerate(words):
      self.word_index[word] = idx
      max_id += 1
    self.word_index['eos'] = max_id
    self.word_index['oov'] = max_id+1
    f = open('word_index.pkl', 'w')
    pickle.dump(self.word_index, f)
    f.close()
    print '***********word index count ', max_id
