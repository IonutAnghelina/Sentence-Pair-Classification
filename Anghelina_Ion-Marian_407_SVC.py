#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


import numpy as np
import pandas as pd 
import json
import re
import string
import xgboost as xgb
from tqdm import tqdm
from keras.preprocessing.text import Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from sklearn.decomposition import SparsePCA, TruncatedSVD, PCA
from sklearn.svm import SVC,LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score,make_scorer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from focal_loss.focal_loss import FocalLoss
import nltk
from sklearn.preprocessing import StandardScaler
from nltk.stem.snowball import SnowballStemmer
from sklearn.metrics import confusion_matrix
import seaborn
from sklearn.metrics import classification_report

# In[2]:


### Used for transforming numbers of up to 4 digits into words, before finding the num2words library. Useless, anyway

# def numTransform(x):
    
#     digitToWords = {0:'zero',1:"unu",2:"doi",3:"trei",4:"patru",5:"cinci",6:"șase",7:"șapte",8:"opt",9:"nouă"}
#     smallNumToWords={10:"zece",11:"unsprezece",12:"doisprezece",13:"treisprezece",14:"paisprezece",15:"cincisprezece",16:"șaisprezece",17:"șaptesprezece",18:"optsprezece",19:"nouăsprezece"}
#     x = int(x)
#     if int(x) in list(range(0,10)):
#         return digitToWords[x]
#     elif x in list(range(10,20)):
#         return smallNumToWords[x]
#     elif x in list(range(20,30)):
#         if x==20:
#             return "douăzeci"
#         else:
#             return "douăzeci și " + digitToWords[x%10]
#     elif x in list(range(30,100)):
#         if x%10==0:
#             return digitToWords[x//10]+"zeci"
#         else:
#             return digitToWords[x//10]+"zeci" + " și " + digitToWords[x%10]
#     elif x in list(range(100,200)):
#         return "o sută " + numTransform(x%100)
#     elif x in list(range(200,300)):
#         return "două sute " + numTransform(x%100)
#     elif x in list(range(300,1000)):
#         return digitToWords[x//100] + " sute " + numTransform(x%100)
#     elif x in list(range(1000,2000)):
#         return"o mie " + numTransform(x%1000)
#     elif x in list(range(2000,3000)):
#         return "două mii " + numTransform(x%1000)
#     elif x in list(range(3000,10000)):
#         return  numTransform(x//1000) + " mii " + numTransform(x%1000) 
    
#     return str(x)
        


# In[3]:


# def isNumber(x):
    
#     for ch in list(str(x)):
#         if ch not in "0123456789":
#             return False 
        
#     return True


# In[4]:


# print(num2words(42, lang='ro'))


# In[5]:


### Old preprocessing, worse result

# def preproc(text):
    
   
     #text = text.replace("[0-9]+","")
    
    
#     #tokens = []
    
#    # for ch in "0123456789":
#     #    text = text.replace(ch,f" {ch} ")
        
    
#     #digitsToWords = {'0':'zero',"1":"unu","2":"doi","3":"trei","4":"patru","5":"cinci","6":"șase","7":"șapte","8":"opt","9":"nouă"}
    
#     #badSpaces = ['\xad','\x80','\x83','\x93','\x96','\x9c','\xa0','\u2009','\u200a','\u200b','\u200e','\u200f','\u202c','\u202f','\u2060','\uf03a','\uf0a7','\uf0be','\uf0e0']
    
#     #for x in badSpaces:
#     #    text = text.replace(x," ")
        
#     #text = text.replace("ṣ Ṭ ṭ")
    
# #     text = text.replace("ṣ","ș")
# #     text = text.replace("Ṭ","Ț")
# #     text = text.replace("ṭ",'ț')
# #     text = text.replace("ˮ","\"")
# #     text = text.replace("`","\'")

# #     text = text.replace("Ş","Ș")
# #     text = text.replace("ş","ș")
# #     text = text.replace("Ţ","Ț")
# #     text = text.replace("ţ","ț")
# #     text = text.replace("\n"," ")
    
# #     text = text.replace("‘","\'")
# #     text = text.replace("’","\'")
    
# #     for x in ['“','”', '„', '‟']:
# #         text = text.replace(x,"\"")
    
# #     text = text.replace("&nbsp;"," ")
    
# #     text = text.replace("("," ( ")
# #     text = text.replace(")"," ) ")
# #     text = text.replace("-"," - ")
# #     text = text.replace("\t",' ')
# #     text = text.replace("."," . ")
# #     text = text.replace(","," ")
# #     text = text.replace("?"," ? ")
# #     text = text.replace("!"," ! ")
    
# #     text = text.replace("(","")
# #     text = text.replace(")","")
    
    
# #     text = text.replace("-"," - ")
    
# #     #text = text.replace("² ³ ´ µ · ¸ ¹ º » ¼ ½ ¾")
    
#     tokens = text.split()
#     #print(tokens)
#     actualTokens = []
#     for x in tokens:
#         #print(x)
#         if isNumber(x):
#             actualTokens+=num2words(x,lang='ro').split()
#         else:
#             actualTokens.append(x)
            
            
#     text = " ".join(actualTokens)
#     return text


# In[6]:


stemmer = SnowballStemmer("romanian") # The stemmer used

allChars = set() #A list of all the characters

def custom_tokenizer(text):


    text = text.replace("²"," putere 2 ") # Replacing exponents
    text = text.replace("³"," putere 3 ")
    text = text.replace("¹"," putere 1 ")
    text = text.replace("¼"," 1 / 4 ") # Replacing fractions
    text = text.replace("½"," 1 / 2 ")
    text = text.replace("¾"," 3 / 4 ")
    text = text.replace("&nbsp"," ") #Replacing HTML sequence
    
    tokens = []
    
    text = text.split()
    for x in text:
        tokens.append(x) #stemming, left here but does not improve the model whatsoever
        
    #if len(tokens)==1:
    #    print("CEVA")
    #    return ['[BL]']
    #for token in text:
      #  tokens.append(token.lemma_)
    #random.shuffle(tokens)
    return " ".join(tokens)


# In[ ]:





# In[ ]:





# In[ ]:





# In[7]:


# print(preproc("Ana are 3 mere ][][/]"))


# In[8]:


train_file = open("./train.json","r",encoding='utf-8') 
#loading the training data
train_data = json.load(train_file)
valid_file = open("./validation.json","r",encoding='utf-8')
valid_data = json.load(valid_file)
#loading the validation data
test_file = open("./test.json","r",encoding='utf-8')
test_data = json.load(test_file)
#loading the test data


# In[9]:


test_file = open("./test.json","r",encoding='utf-8') 
#there are some times in my code where i need to only reload the test data
test_data = json.load(test_file)


# In[10]:


first_sentences_train = [] #second sentences are not used now, previously tried to concatenate vectors for each sentence AFTER vectorizing
second_sentences_train = []
labels_train = []
guids = []
for x in train_data:
    first_sentences_train.append(x['sentence1']+' [SEPTOK] '+x['sentence2']) 
    #concatenated train sentences
    second_sentences_train.append(x['sentence2']+' [SEPTOK] ' + x['sentence2'])
    labels_train.append(x['label'])
    guids.append(x['guid'])
    allChars=(allChars|set(list(x['sentence1']+x['sentence2'])))
    


# In[11]:


first_sentences_valid = []
second_sentences_valid = []
labels_valid = []
guids = []
for x in valid_data:
    first_sentences_valid.append(x['sentence1']+' [SEPTOK] ' + x['sentence2'])
    #concatenated validation sentences
    second_sentences_valid.append(x['sentence2'])
    labels_valid.append(x['label'])
    guids.append(x['guid'])
    allChars=(allChars|set(list(x['sentence1']+x['sentence2'])))
    
#     first_sentences_train.append(x['sentence1']+' [SEPTOK] '+x['sentence2']) #trained on the validation set prior to the submission
#     second_sentences_train.append(x['sentence2']+' [SEPTOK] ' + x['sentence2'])
#     labels_train.append(x['label'])
#     guids.append(x['guid'])
#     allChars=(allChars|set(list(x['sentence1']+x['sentence2'])))
    


# In[12]:


" ".join(sorted(list(allChars))) 
#all the distinct characters in the set


# In[13]:


vectorizer = TfidfVectorizer(lowercase=False,analyzer = 'char',ngram_range=(1,5),norm='l2',preprocessor = custom_tokenizer)
#my vectorizer with the custom parameters provided in the documentation
vectorizer.fit(first_sentences_train)

tf_first_sentences_train=vectorizer.transform(first_sentences_train)





#print(tf_first_sentences_train.shape)
#print(tf_second_sentences_train.shape)
#print(tf_first_sentences_train2.shape)
#print(tf_second_sentences_train2.shape)



# In[ ]:





# In[ ]:





# In[14]:


model = LinearSVC(verbose=2,penalty='l2',C=50,class_weight = 'balanced') 
#The LinearSVC classifier with the parameters from the documentation


# In[15]:


model.fit(tf_first_sentences_train,labels_train)


# In[16]:


tf_first_sentences_valid=vectorizer.transform(first_sentences_valid)
#vectorizing the validation sentences


# In[17]:


valid_data = tf_first_sentences_valid


# In[18]:


validation_preds = model.predict(valid_data)


# In[28]:


print(f1_score(labels_valid,validation_preds,average = 'macro'))
# Testing on the validation set


# In[29]:


print(f1_score(validation_preds,labels_valid,average = None))
# Testing on the validation set for each class


# In[27]:





# In[32]:


confusionMatrix = confusion_matrix(labels_valid,validation_preds)
seaborn.heatmap(confusionMatrix,annot=True)
plt.show()

#Computing and printing the confusion_matrix


# In[33]:



print(classification_report(labels_valid,validation_preds))

#Full metrics about the model


# In[ ]:





# In[21]:


test_file = open("./test.json","r",encoding='utf-8')
test_data = json.load(test_file)


# In[22]:


first_sentences_test = []
second_sentences_test = []
labels_test = []
guids = []
for x in test_data:
    first_sentences_test.append(x['sentence1']+' [SEPTOK] ' + x['sentence2'])
    #Reading test data
    second_sentences_test.append(x['sentence2'])
    labels_test.append(0)
    guids.append(x['guid'])


# In[23]:


tf_first_sentences_test=vectorizer.transform(first_sentences_test)
print(tf_first_sentences_test.shape)
#Vectorizing test data


# In[ ]:





# In[24]:


predictions = model.predict(tf_first_sentences_test) 
#Predicting on the test data


# In[25]:


with open("submission_SVC.csv","w") as fout: 
    #Building the submission
    
    print("guid,label",file=fout)
    for x,y,z in zip(guids,predictions,first_sentences_test):
        print(f"{x},{y}",file=fout)
        


# In[26]:


print(predictions)


# In[ ]:





# In[ ]:





# In[ ]:




