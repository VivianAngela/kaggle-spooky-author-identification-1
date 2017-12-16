import numpy as np
import argparse
import os
import tensorflow as tf
import sys
import random
import operator
import json
import keras
import csv
from time import time
from keras.models import Sequential
from keras.layers import Dense, LSTM, Activation, Dropout, TimeDistributed, RepeatVector, Embedding
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard
from keras.backend.tensorflow_backend import set_session
from keras.optimizers import Adam
from keras import backend as K
from utils import *
from data import Data, Embeddings
from model import get_model, train, predict
from config import *

parser = argparse.ArgumentParser()
parser.add_argument("--mode", required=True)
parser.add_argument("--model_file")
args = parser.parse_args()

tfconfig = tf.ConfigProto()
tfconfig.gpu_options.per_process_gpu_memory_fraction = 1
set_session(tf.Session(config=tfconfig))

if args.mode == 'train':
  print "parsing input data"
  data = Data()

  print "loaded training data"
  print "initiating model construction"

  model = get_model(data)
  keras.utils.plot_model(model, to_file='model.png')
  train(model, data)
  
elif args.mode == 'predict':
  data = Data()
  model = get_model(data)
  weights= args.model_file
  if weights != None and weights.strip() != "":
    model.load_weights(weights)
    print "model loaded"
  else:
    print "No model to load"
  
  with open('test.csv') as f, open('submission.csv', 'w') as of:
    reader = csv.reader(f)
    writer = csv.writer(of)
    records = [r for r in reader]
    i=0
    for record in records[1:]:
      print 'predicting ' + str(i+1) + ' of 8393 records'
      i+=1
      id = record[0]
      text = record[1]
      pred = predict(model, text, data.embeddings)
      eap =  "{:.16f}".format(pred[0,1])
      hpl =  "{:.16f}".format(pred[0,2])
      mws =  "{:.16f}".format(pred[0,0])
      writer.writerow([id,eap,hpl,mws])
