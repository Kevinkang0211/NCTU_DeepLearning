#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import nltk
#nltk.download()

from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt


from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

import string
import re


# In[ ]:





# In[ ]:





# In[2]:


from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence


# In[3]:


x_train = pd.read_csv("train.csv")

y_train = np.array(pd.get_dummies(x_train.Category))
#labels = pd.factorize(x_train.Category)
x_train = x_train.drop(['Id','Category'],axis=1)

x_test = pd.read_csv("test.csv")
x_test = x_test.drop(['Id'],axis=1)


# In[4]:


seqlen_train = x_train['Title'].apply(lambda x : len(x.split()))
seqlen_test = x_test['Title'].apply(lambda x : len(x.split()))


# In[5]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[6]:


#看句子大概的長度範圍

sns.set_style('darkgrid')
plt.figure(figsize=(8,4))
sns.histplot(seqlen_train)
sns.histplot(seqlen_test)


# In[7]:



'''
ps = PorterStemmer()

titles = x_train['Title'].copy()
stop_words = set(stopwords.words('english'))

for i in range(len(titles)):
    #讓標題轉成小寫
    titles[i] = titles[i].lower()
    
    #去除標點符號 Removing punctuations Using regex
    titles[i] = re.sub(r'[^\w\s]', '', titles[i])
    
    
    #Remove whitespaces, remove leading and ending spaces
    titles[i] = titles[i].strip()
    
    #用NLTK token文字
    titles[i] = word_tokenize(titles[i])
    
    #去除stop word
    #titles[i] = [i for i in titles[i] if not i in stop_words]
    
x_train['Title'] = titles

#stemming處理
for i in range(len(x_train['Title'])):
    templist=[]
    for w in x_train['Title'][i]:
        templist.append(ps.stem(w))
    x_train['Title'][i] = templist




    #x_test 前處理
    
titles = x_test['Title'].copy()

for i in range(len(titles)):
    #讓標題轉成小寫
    titles[i] = titles[i].lower()
    
    #去除標點符號 Removing punctuations Using regex
    titles[i] = re.sub(r'[^\w\s]', '', titles[i])
    
    
    #Remove whitespaces, remove leading and ending spaces
    titles[i] = titles[i].strip()
    
    #用NLTK token文字
    titles[i] = word_tokenize(titles[i])
    
    #去除stop word
    #titles[i] = [i for i in titles[i] if not i in stop_words]
    
x_test['Title'] = titles

#stemming處理
for i in range(len(x_test['Title'])):
    templist=[]
    for w in x_test['Title'][i]:
        templist.append(ps.stem(w))
    x_test['Title'][i] = templist
   
'''


# text preprocessing 在transformer暫時不要做

# In[7]:


x_train


# In[8]:


x_test


# In[ ]:





# In[ ]:





# In[9]:



token = Tokenizer(num_words=6000)  

#使用Tokenizer模組建立token，建立一個字典


# In[10]:


token.fit_on_texts(x_train['Title'])
token.fit_on_texts(x_test['Title'])


# In[11]:


x_train_seq = token.texts_to_sequences(x_train['Title'])
x_test_seq = token.texts_to_sequences(x_test['Title'])

x_train = sequence.pad_sequences(x_train_seq,maxlen=15)
x_test = sequence.pad_sequences(x_test_seq,maxlen=15)


# In[ ]:





# In[12]:


from keras.models import Sequential
from keras.layers.core import Dense,Dropout,Activation,Flatten
from keras.layers.embeddings import Embedding
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers


# In[41]:


'''import os


embeddings_index = {}
f = open(os.path.join('C:/Users/kevin/glove.6B', 'glove.6B.300d.txt'),encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))'''


# In[42]:


'''word_index = token.word_index

embedding_matrix = np.zeros((len(word_index) + 1, 300))

for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector'''


# In[43]:


'''embedding_matrix.shape'''


# In[44]:





# In[13]:


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.3):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="tanh"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


# In[14]:


class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        #self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim,weights=[testembedding],
                                                                                    input_length = 15,
                                                                                     trainable=True)
        
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        
        x = self.token_emb(x)
        return x + positions
    
    


# In[ ]:





# ## HW 1.3 Transformer Model Setting

# Hyperparameter of Transformer:
# 
# -Attention head 設為3
# 
# -Transformer block裡面的隱藏層的layer size設為16
# 
# -Transfromer block後面接一個 64nodes 的隱藏層，activation function 經實驗過後設為tanh表現最好

# In[54]:


'''#設定EARLY STOP機制 監控loss變化
callback = keras.callbacks.EarlyStopping(monitor='loss', patience=1)'''


# In[65]:


'''
vocab_size,embed_dim = embedding_matrix.shape
maxlen = 15  # Only consider the first 200 words of each movie review
num_heads = 3  # Number of attention heads
ff_dim = 16  # Hidden layer size in feed forward network inside transformer

testembedding = embedding_matrix.copy()

inputs = layers.Input(shape=(maxlen,))

embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)

x = embedding_layer(inputs)

transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)

x = transformer_block(x)

x = layers.GlobalAveragePooling1D()(x)
x = layers.Dropout(0.3)(x)


x = layers.Dense(64 ,activation="tanh")(x)
x = layers.Dropout(0.3)(x)


outputs = layers.Dense(5, activation="softmax")(x)

model = keras.Model(inputs=inputs, outputs=outputs)'''


# In[66]:


'''
optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False)

loss = tf.keras.losses.CategoricalCrossentropy()  # categorical = one-hot
acc = tf.keras.metrics.CategoricalAccuracy('accuracy')

model.compile(optimizer=optimizer, loss=loss, metrics=[acc])


train_history = model.fit(x_train, y_train, 
                          batch_size=32, 
                          epochs=80,
                          verbose=2,
                          callbacks=[callback],
                          #validation_split=0.1
)'''


# In[ ]:





# In[67]:


'''
y_predict = model.predict(x_test, verbose=0)

y_predict = np.argmax(y_predict,axis=1)

classes = ['business','entertainment','politics','sport','tech']

result =[]
for i in range(len(y_predict)):
    result.append([i,classes[y_predict[i]]])
    
final_result = pd.DataFrame(data=result, index=None, columns=['Id','Category'], dtype=None, copy=False)

final_result.to_csv(r'output.csv', index = False)'''


# In[ ]:





# In[68]:


'''model.save('transformer_model')'''


# In[15]:


reconstructed_model = keras.models.load_model("transformer_model")


# In[16]:



y_predict = reconstructed_model.predict(x_test, verbose=0)

y_predict = np.argmax(y_predict,axis=1)

classes = ['business','entertainment','politics','sport','tech']

result =[]
for i in range(len(y_predict)):
    result.append([i,classes[y_predict[i]]])
    
final_result = pd.DataFrame(data=result, index=None, columns=['Id','Category'], dtype=None, copy=False)

final_result.to_csv(r'309706033_submission_transformer.csv', index = False)


# In[ ]:




