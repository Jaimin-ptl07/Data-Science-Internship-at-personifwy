#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
import numpy as py


# In[2]:


with open("train_qa-220120-145526.txt","rb") as fp:
    train_data = pickle.load(fp)


# In[3]:


train_data


# In[4]:


with open("test_qa-220120-145430.txt","rb") as fp:
    test_data = pickle.load(fp)


# In[5]:


test_data


# In[6]:


' '.join(train_data[0][2])


# In[7]:


len(train_data)


# In[8]:


#set up vocabulary
vocab = set()


# In[9]:


all_data = train_data + test_data


# In[10]:


for story, question, answer in all_data:
    vocab = vocab.union(set(story))
    vocab = vocab.union(set(question))


# In[11]:


vocab.add("yes")
vocab.add("no")


# In[12]:


vocab


# In[13]:


vocab_len = len(vocab)+1


# In[14]:


max_story_len = max([len(data[0]) for data in all_data])
max_story_len


# In[15]:


max_question_len = max([len(data[1]) for data in all_data])
max_question_len


# In[16]:


#vectorize
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer


# In[17]:


tokenizer = Tokenizer()
tokenizer.fit_on_texts(vocab)
tokenizer.word_index


# In[18]:


train_story_text = []
train_question_text = []
train_answer_text = []

for story, question, answer in train_data:
    train_story_text.append(story)
    train_question_text.append(question)


# In[19]:


train_story_seq = tokenizer.texts_to_sequences(train_story_text)
train_story_seq


# In[20]:


train_story_text


# In[25]:


def vectorize_stories(data, word_index = tokenizer.word_index,
                     max_story_len = max_story_len, max_question_len = max_question_len):

    X= []
    Xq = []
    Y = []

    for story, question, answer in data:
        x = [word_index[word.lower()] for word in story]
        xq = [word_index[word.lower()] for word in question]
        y=np.zeros(len(word_index)+1)
        y[word_index[answer]]=1

        X.append(x)
        Xq.append(xq)
        Y.append(y)
    
    return(pad_sequences(X, maxlen = max_story_len),
          pad_sequences(Xq, maxlen = max_question_len),
          np.array(Y))


# In[26]:


inputs_train, question_train, answers_train = vectorize_stories(train_data)
inputs_test, question_test, answers_test = vectorize_stories(test_data)


# In[ ]:


inputs_test


# In[ ]:


tokenizer.word_index['yes']


# In[ ]:


from keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding
from keras.layers import Input, Activation, Dense, Permute, Dropout, add, dot, concatenate, LSTM


# In[ ]:


input_sequence  = input((max_story_len,))
question_sequence = input((max_question_len,))


# In[ ]:


#input encoder m
input_encoder_m = Sequential()
input_encoder_m.add(Embedding(input_dim = vocab_len, output_dim = 64))
input_encoder_m.add(Dropout(0.3))


# In[ ]:


#input encoder c
input_encoder_c = Sequential()
input_encoder_c.add(Embedding(input_dim = vocab_len, output_dim = max_question_len))
input_encoder_c.add(Dropout(0.3))


# In[ ]:


#question encoder
question_encoder = Sequential()
question_encoder.add(Embedding(input_dim = vocab_len, output_dim = max_question_len, input_length = max_question_len))
question_encoder.add(Dropout(0.3))


# In[ ]:


#Encode the sequences
input_encoded_m = input_encoder_m(input_sequence)
input_encoded_c = input_encoder_c(input_sequence)
question_encoded = question_encoder(question)


# In[ ]:


match = dot([input_encoded_m,question_encoded], axes =(2,2))
match = Activation('softmax')(match)


# In[ ]:


response = add([match,input_encoded_c])
response = Permute((2,1))(response)


# In[ ]:


#concatenate
answer = concatenate([response,question_encoded])


# In[ ]:


answer


# In[ ]:


answer = LSTM(32)(answer)


# In[ ]:


answer = Dropout(0.5)(answer)
answer = Dense(vocab_len)(answer)


# In[ ]:


answer = Activation('softmax')(answer)


# In[ ]:


model = model([input_sequence, question], answer)
model.complie(optimizer = 'rmsprop', loss = 'catagorical_crossentropy', metrics = ['accuracy'])


# In[ ]:


model.summary()


# In[ ]:


history = model.fit([inputs_train, question_train], answer_train,
                    batch_size = 32, epochs =20, validation_data([inputs_data, question_test], answer_test)
                   )


# In[ ]:


import matplotlib.pyplot as plt
print(history,history.keys())
plt.plot(history.history['accuracy'])


# In[ ]:


plt.title('Model Accuracy')
plt.xlabel('Accuracy')
plt.ylabel('epochs')


# In[ ]:


#save
model.save("chatbot_model")


# In[ ]:


#evaluation based on testset
model.load.weights("chatbot_model")


# In[ ]:


pred_results = model.predict(([intput_test,question_test]))


# In[ ]:


story = ' '.join(word for word in test_data[0][0])
story


# In[ ]:


question = ' '.join(word for word in test_data[0][1])
question


# In[ ]:


test_data[0][2]


# In[ ]:


val_max = py.argmax(pred_results[0])
for key,val in tokenizer.word_index.items():
    if val==val_max:
        k=key
        
print("The predicted value is :",k)
print("The probability of certainity is:", pred_results[0][val_max])


# In[ ]:




