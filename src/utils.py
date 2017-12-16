import numpy as np
import re
from keras.preprocessing.text import text_to_word_sequence
import spacy

nlp = spacy.load('en')

def get_embedding_input(text, max_len, input_shape, embedding):
  text += ' __eos__'
  text = re.sub(r"([^\w\s])", r" \1 ", text)
  tokens = text_to_word_sequence(text, lower=True)
  tokens = [t for t in tokens[:max_len]]
  embedding_input = np.zeros(input_shape, dtype='int')
  for idx, word in enumerate(tokens):
    if idx >= max_len:
      break
    if embedding.word_index.has_key(word):
      t = embedding.word_index[word]
    else:
      print word
      t = embedding.word_index['oov']
    embedding_input[idx] = t
  return embedding_input

tags = ["ADJ", "ADP", "ADV", "AUX", "CONJ", "CCONJ",
        "DET", "INTJ", "NOUN", "NUM", "PART", "PRON",
        "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X",
        "SPACE"]
ntags = len(tags)

def pos_word_input(pos_tag):
  one_hot = []
  tag_index = tags.index(pos_tag)
  for i in range(ntags):
    if i == tag_index:
      one_hot.append(1)
    else:
      one_hot.append(0)
  return np.asarray(one_hot, dtype='int32')

def tags_input(text, seq_length):
  doc = nlp(unicode(text, 'utf-8'))
  l = len(doc)
  inp = []
  for i in range(seq_length):
    if i >= l:
      inp.append(np.zeros((19,)))
    else:
      t = pos_word_input(doc[i].pos_)
      inp.append(pos_word_input(doc[i].pos_))
  return np.asarray(inp, dtype='int32')

