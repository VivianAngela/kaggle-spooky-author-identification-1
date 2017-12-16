import keras
import os
import numpy as np
import tensorflow as tf
from time import time
from keras.layers import LSTM, Embedding, Input, Dense, Dropout, Bidirectional, BatchNormalization, Concatenate, TimeDistributed, Lambda, GRU, Layer
from keras.models import Model
from keras.utils.vis_utils import plot_model
from keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard, EarlyStopping
from keras.optimizers import Adam
from keras import backend as K
from utils import * 
from config import *

class Attention(Layer):
  # custom attention referred from Sujit Pal's attention https://github.com/sujitpal/eeap-examples
  def build(self, input_shape):
    self.W = self.add_weight(name="W_{:s}".format(self.name), shape=(input_shape[-1],1), initializer='normal')
    self.b = self.add_weight(name="b_{:s}".format(self.name), shape=(input_shape[1],1), initializer='zeros')
    super(Attention, self).build(input_shape)

  def call(self, x, mask=None):
    """ from Sujit Pal's article
        e_t = tanh(Uc + Wh_t + b)
        a_t = softmax(e_t)
        o = sum(a_t * h_t)
    """
    d = K.dot(x, self.W)
    et = K.squeeze(K.relu(d + self.b), axis=-1)
    at = K.softmax(et)
    if mask is not None:
      at *= K.cast(mask, K.floatx())
    atx = K.expand_dims(at, axis=-1)
    o = x * atx
    return K.sum(o, axis=1)
  
  def compute_mask(self, input, input_mask=None):
    return None
  
  def compute_output_shape(self, input_shape):
    return (input_shape[0], input_shape[-1])
  

def get_model(data):

  if dual_embedding:
    text_input1 = Input(shape=(seq_length,),
                            name='text1',
                            dtype='int32')

    text_embedding1 = Embedding(vocab_size+2, embedding_dim, 
                                    input_length=seq_length,
                                    mask_zero=True,
                                    name='text_embedding1',
                                    weights=[data.embeddings.embedding_matrix],
                                    trainable=False) (text_input1)
    text_input2 = Input(shape=(seq_length,),
                            name='text2',
                            dtype='int32')

    text_embedding2 = Embedding(vocab_size+2, embedding_dim,
                                    input_length=seq_length,
                                    mask_zero=True,
                                    name='text_embedding2',
                                    weights=[data.embeddings.embedding_matrix],
                                    trainable=True) (text_input2)
    enc = Bidirectional(GRU(lstm_size, name='enc', return_sequences=True))
    text_enc1 = enc(text_embedding1)
    text_enc2 = enc(text_embedding2)
    attention = Attention()
    text_att1 = attention(text_enc1)
    text_att2 = attention(text_enc2)
    
    #pos_input = Input(shape=(), name='pos', dtype='int32')
    concat = Concatenate()([text_att1, text_att2])
    drop1 = Dropout(0.2)(concat)
    bn1 = BatchNormalization()(drop1)
    dense1 = Dense(dense_size, activation='relu')(bn1)
    drop2 = Dropout(0.2)(dense1)
    bn2 = BatchNormalization()(drop2)
    """
    dense2 = Dense(dense_size, activation='relu')(bn2)
    drop3 = Dropout(0.2)(dense2)
    bn3 = BatchNormalization()(drop3)
    dense3 = Dense(dense_size, activation='relu')(bn3)
    drop4 = Dropout(0.2)(dense3)
    bn4 = BatchNormalization()(drop4)
    """
    output = Dense(3, activation='softmax')(bn2)
    model = Model(inputs=[text_input1,text_input2], outputs=output)
    #model = Model(inputs=text_input1, outputs=output)

  else:
    trainable_embedding = embedding_type == 'trainable'

    text_input = Input(shape=(seq_length,),
                           name='text',
                           dtype='int32')

    text_embedding = Embedding(vocab_size+2,
                                   embedding_dim,
                                   input_length=seq_length,
                                   mask_zero=True,
                                   name='embedding',
                                   weights=[data.embeddings.embedding_matrix],
                                   trainable=trainable_embedding) (text_input)

    text_outputs = Bidirectional(LSTM(lstm_size, name='encoder1', return_sequences=True))(text_embedding)
    att = Attention()(text_outputs)
    pos_input = Input(shape=(seq_length,),
                      name='pos',
                      dtype='int32')
    pos_enc = LSTM(lstm_size, name='pos_enc')(pos_input)
    concat = Concatenate([att, pos_enc])
    drop1 = Dropout(0.2)(concat)
    bn1 = BatchNormalization()(drop1)  
    dense1 = Dense(dense_size, activation='relu')(bn1)
    drop2 = Dropout(0.2)(dense1)
    bn2 = BatchNormalization()(drop2)
    dense2 = Dense(dense_size, activation='relu')(bn2)
    drop3 = Dropout(0.2)(dense2)
    bn3 = BatchNormalization()(drop3)
    output = Dense(3, activation='softmax', name='classifier_output')(bn3)
    model = Model(inputs=text_input, outputs=output)

  return model

def train(model, data):
  
  optimizer = Adam(lr=0.005)
  model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
  print model.summary()

  checkpoint = ModelCheckpoint('weights-improvement-{epoch:02d}-{loss:.2f}.hdf5', verbose=1)
  csv_logger = CSVLogger('log.csv', append=True, separator='\t')
  tensorboard = TensorBoard(log_dir='logs'.format(time()))
  earlystop = EarlyStopping(monitor='val_loss', patience=4)
  callbacks_list = [checkpoint, csv_logger, tensorboard]

  """
  model.fit_generator(data.input_generator(),
                      steps_per_epoch = data.train_count/batch_size,
                      epochs = epochs,
                      callbacks = callbacks_list,
                      validation_data = data.input_generator(False),
                      validation_steps = 100,
                      workers = 1)
  """
  model.fit([np.asarray(data.train_text_data),np.asarray(data.train_pos_data)], np.asarray(data.train_targets), batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks=callbacks_list)
  #model.fit([np.asarray(data.train_data),np.asarray(data.train_data)], np.asarray(data.train_targets), batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks=callbacks_list)

def predict(model, text, embedding):
  
  I = []

  t = get_embedding_input(text, seq_length, (seq_length,), embedding)
  I.append(t)

  print 'going to predict'
  #pred = model.predict(np.asarray(I, dtype='float32'), batch_size=1)
  pred = model.predict([np.asarray(I, dtype='float32'),np.asarray(I, dtype='float32')], batch_size=1)
  
  return pred
