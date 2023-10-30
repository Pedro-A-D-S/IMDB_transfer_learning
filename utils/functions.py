import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import pandas as pd

from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.layers import (
    Embedding,
    LSTM,
    Bidirectional,
    Dense,
    Dropout
)

from transformers import BertTokenizer
from transformers import TFBertForSequenceClassification
from transformers import TFBertModel

from sklearn.model_selection import train_test_split


def encode_pad_transform(sample):
    encoded = imdb_encoder.encode(sample.numpy())
    pad = sequence.pad_sequences([encoded], padding='post', 
                                 maxlen=150)
    return np.array(pad[0], dtype=np.int64)  


def encode_tf_fn(sample, label):
    encoded = tf.py_function(encode_pad_transform, 
                                       inp=[sample], 
                                       Tout=(tf.int64))
    encoded.set_shape([None])
    label.set_shape([])
    return encoded, label

def build_model_bilstm(vocab_size, embedding_dim, 
                       rnn_units, batch_size, train_emb=False):
  model = tf.keras.Sequential([
    Embedding(vocab_size, embedding_dim, mask_zero=True,
              weights=[embedding_matrix], trainable=train_emb),
    #Dropout(0.25),  
    Bidirectional(tf.keras.layers.LSTM(rnn_units, return_sequences=True, 
                                      dropout=0.5)),
    Bidirectional(tf.keras.layers.LSTM(rnn_units, dropout=0.25)),
    Dense(1, activation='sigmoid')
  ])
  return model

def bert_encoder(review):
    txt = review.numpy().decode('utf-8')
    encoded = tokenizer.encode_plus(txt, add_special_tokens=True, 
                                    max_length=150, pad_to_max_length=True, 
                                    return_attention_mask=True, 
                                    return_token_type_ids=True,
                                    truncation=True)
    return encoded['input_ids'], encoded['token_type_ids'], \
           encoded['attention_mask']


def example_to_features(input_ids,attention_masks,token_type_ids,y):
  return {"input_ids": input_ids,
          "attention_mask": attention_masks,
          "token_type_ids": token_type_ids},y