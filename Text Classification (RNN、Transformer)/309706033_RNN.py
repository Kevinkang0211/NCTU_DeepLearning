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


# ## 1.1 Text Preprocessing 部分

# 1.本次文字前處裡採用NLTK套件，作為Tokenizer的工具；且針對空白字元，我們採取忽略不tokenize它們
# 
# 2.加入special token的原因是因為每個標題長度要一致，所以我們要用 PAD 來填充過短的標題；另外如果字詞沒在字典裡出現過，就用 UNK 的替代它
# 
# 3.Text Preprocessing的程序為: 
# 
# 讓標題轉為小寫 --> 利用Regex去除標點符號 --> 去除white spaces -->用NLTK token文字 --> 去除stop word --> 用NLTK stemming 文字

# In[ ]:





# In[4]:




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
    titles[i] = [i for i in titles[i] if not i in stop_words]
    
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
    titles[i] = [i for i in titles[i] if not i in stop_words]
    
x_test['Title'] = titles


#stemming處理
for i in range(len(x_test['Title'])):
    templist=[]
    for w in x_test['Title'][i]:
        templist.append(ps.stem(w))
    x_test['Title'][i] = templist


# In[5]:


x_train


# In[6]:


x_test


# In[7]:



token = Tokenizer(num_words=6000)  

#使用Tokenizer模組建立token，建立一個6000字的字典


# In[8]:


token.fit_on_texts(x_train['Title'])
token.fit_on_texts(x_test['Title'])


# In[9]:


x_train_seq = token.texts_to_sequences(x_train['Title'])
x_test_seq = token.texts_to_sequences(x_test['Title'])

x_train = sequence.pad_sequences(x_train_seq,maxlen=20)
x_test = sequence.pad_sequences(x_test_seq,maxlen=20)


# In[ ]:





# ## HW1.2 RNN

# 【Word Embedding】
# 
# -這次使用Pre-trained的 word embedding是使用 Glove.6B.200d檔
# 
# -會使用pre-trained的word embedding的原因是因為，這些像是glove、w2v、fasttext都是研究機構或是企業已經將好幾百萬、千萬的文章、文字都看過，並編成的word vector，所以有點像是站在巨人的肩膀上一樣，可以將這些已經學過、訓練過的文字向量作為我們預設的向量，而不是一開始隨機預設然後再慢慢學習。
# 
# -將每個標題轉為Keras可讀的形式，在截長補短padding到長度為20
# 
# 【Model Design】
# 
# -本次任務使用了 GRU 作為預測模型
# 
# -模型架構為: 64個神經元的GRU -> 32個神經元的隱藏層 採用relu作為activation function -> 最後一層為softmax 5個神經元的輸出
# 
# 
# 

# In[10]:


'''import os


embeddings_index = {}
f = open(os.path.join('C:/Users/kevin/glove.6B', 'glove.6B.200d.txt'),encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))'''


# In[65]:


'''word_index = token.word_index

embedding_matrix = np.zeros((len(word_index) + 1, 200))

for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector'''


# In[10]:


from keras.models import Sequential
from keras.layers.core import Dense,Dropout,Activation,Flatten
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import GRU
from keras.layers import Bidirectional


# In[73]:


'''modelGRU = Sequential() #建立模型

modelGRU.add(Embedding(len(word_index) + 1,
                            200,
                            weights=[embedding_matrix],
                            input_length=20,
                            trainable=False))


modelGRU.add(Dropout(0.2)) #隨機在神經網路中放棄神經元，避免overfitting

#建立64個神經元的GRU層

modelGRU.add(GRU(64, dropout=0.2, recurrent_dropout=0.2))
modelGRU.add(Dense(units=32,activation='relu'))
modelGRU.add(Dropout(0.2))
#建立一個輸出層
modelGRU.add(Dense(units=5,activation='softmax'))

'''


# In[74]:


#設定EARLY STOP機制 監控loss變化
'''callback = keras.callbacks.EarlyStopping(monitor='loss', patience=3)'''


# In[75]:


'''modelGRU.compile(loss='categorical_crossentropy',
optimizer='adam',
metrics=['accuracy'])

train_history = modelGRU.fit(x_train,y_train,
epochs=50,
batch_size=32,
callbacks=[callback],
verbose=1,
validation_split=0.3)
'''


# In[80]:


'''#將訓練好的模型儲存起來
modelGRU.save('GRU_model')'''


# In[76]:


'''y_predict = modelGRU.predict(x_test, verbose=0)

y_predict = np.argmax(y_predict,axis=1)

classes = ['business','entertainment','politics','sport','tech']

result =[]
for i in range(len(y_predict)):
    result.append([i,classes[y_predict[i]]])
    
final_result = pd.DataFrame(data=result, index=None, columns=['Id','Category'], dtype=None, copy=False)

final_result.to_csv(r'output.csv', index = False)'''


# In[11]:


#將先前儲存的模型載入

reconstructed_model = keras.models.load_model("GRU_model")


# In[12]:



y_predict = reconstructed_model.predict(x_test, verbose=0)

y_predict = np.argmax(y_predict,axis=1)

classes = ['business','entertainment','politics','sport','tech']

result =[]
for i in range(len(y_predict)):
    result.append([i,classes[y_predict[i]]])
    
final_result = pd.DataFrame(data=result, index=None, columns=['Id','Category'], dtype=None, copy=False)

final_result.to_csv(r'309706033_submission_RNN.csv', index = False)


# In[ ]:




