#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[ ]:





# In[ ]:





# In[1]:


import numpy as np
import pandas as pd 
import json
import re
import string
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import seaborn
from sklearn.metrics import classification_report
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
import nltk
from sklearn.preprocessing import StandardScaler
import random
from unidecode import unidecode
from num2words import num2words
from nltk.stem.snowball import SnowballStemmer


# In[ ]:





# In[ ]:





# In[ ]:





# In[2]:


def numTransform(x): #converting to numbers
    
    digitToWords = {0:'zero',1:"unu",2:"doi",3:"trei",4:"patru",5:"cinci",6:"șase",7:"șapte",8:"opt",9:"nouă"}
    smallNumToWords={10:"zece",11:"unsprezece",12:"doisprezece",13:"treisprezece",14:"paisprezece",15:"cincisprezece",16:"șaisprezece",17:"șaptesprezece",18:"optsprezece",19:"nouăsprezece"}
    x = int(x)
    if int(x) in list(range(0,10)):
        return digitToWords[x]
    elif x in list(range(10,20)):
        return smallNumToWords[x]
    elif x in list(range(20,30)):
        if x==20:
            return "douăzeci"
        else:
            return "douăzeci și " + digitToWords[x%10]
    elif x in list(range(30,100)):
        if x%10==0:
            return digitToWords[x//10]+"zeci"
        else:
            return digitToWords[x//10]+"zeci" + " și " + digitToWords[x%10]
    elif x in list(range(100,200)):
        return "o sută " + numTransform(x%100)
    elif x in list(range(200,300)):
        return "două sute " + numTransform(x%100)
    elif x in list(range(300,1000)):
        return digitToWords[x//100] + " sute " + numTransform(x%100)
    elif x in list(range(1000,2000)):
        return"o mie " + numTransform(x%1000)
    elif x in list(range(2000,3000)):
        return "două mii " + numTransform(x%1000)
    elif x in list(range(3000,10000)):
        return  numTransform(x//1000) + " mii " + numTransform(x%1000) 
    
    return str(x)
        


# In[3]:


def isNumber(x): #checking if it's a number
    
    for ch in list(str(x)):
        if ch not in "0123456789":
            return False 
        
    return True


# In[ ]:





# In[4]:


stemmer = SnowballStemmer("romanian") # Stemming step
allChars = set()

def custom_tokenizer(text):


    text = text.replace("²"," putere 2 ")
    #Replacing exponents
    text = text.replace("³"," putere 3 ")
    text = text.replace("¹"," putere 1 ")
    text = text.replace("¼"," 1 / 4 ") 
    #Replacing fraction characters
    text = text.replace("½"," 1 / 2 ")
    text = text.replace("¾"," 3 / 4 ")
    text = text.replace("&nbsp"," ")
    #Replacing HTML characters
    
    tokens = []
    
    text = text.split()
    for x in text:
        #we stem each token
        tokens.append(stemmer.stem(x))
        
    if len(tokens)==1:
        #If a sentence is very short (only 1 word), it gets replace by a special token
        print("CEVA")
        return '[BL]'
    #for token in text:
      #  tokens.append(token.lemma_)
    #random.shuffle(tokens)
    return unidecode(" ".join(tokens)) #We romanize the text and recomponse the tokens


# In[5]:


#print(allChars)


# In[6]:


print(custom_tokenizer("Ana are 3 mere")) 
#a small test


# In[7]:


train_file = open("./train.json","r",encoding='utf-8') 
#Loading the data from the json
train_data = json.load(train_file)
valid_file = open("./validation.json","r",encoding='utf-8')
#loading the validation data
valid_data = json.load(valid_file)
#Loading the test data
test_file = open("./test.json","r",encoding='utf-8')
test_data = json.load(test_file)


# In[8]:


#needed to only run this cell sometimes
test_file = open("./test.json","r",encoding='utf-8')
test_data = json.load(test_file)


# In[9]:


first_sentences_train = []
second_sentences_train = []
labels_train = []
guids = []
for x in train_data:
    first_sentences_train.append(x['sentence1']+x['sentence2']) 
    #Building the training dataset
    second_sentences_train.append(x['sentence2'][:-1])
    labels_train.append(x['label'])
    guids.append(x['guid'])
    allChars=(allChars|set(list(x['sentence1']+x['sentence2'])))
    
#     if x['label'] in [0,1]:
#         first_sentences_train.append(x['sentence2'])
#         second_sentences_train.append(x['sentence1'])
#         labels_train.append(x['label'])
#         guids.append(x['guid'])


# In[10]:


first_sentences_valid = []
second_sentences_valid = []
labels_valid = []
guids = []
for x in valid_data:
    first_sentences_valid.append(x['sentence1']+ x['sentence2']) 
    #Building the validation dataset
    second_sentences_valid.append(x['sentence2'][:-1])
    labels_valid.append(x['label'])
    guids.append(x['guid'])
    allChars=(allChars|set(list(x['sentence1']+x['sentence2'])))
    
#     first_sentences_train.append(x['sentence1']+x['sentence2'])
#     second_sentences_train.append(x['sentence2'][:-1])
#     labels_train.append(x['label'])
#     guids.append(x['guid'])
#     allChars=(allChars|set(list(x['sentence1']+x['sentence2'])))
    
#     if x['label'] in [0,1]:
#         first_sentences_train.append(x['sentence2'])
#         second_sentences_train.append(x['sentence1'])
#         labels_train.append(x['label'])
#         guids.append(x['guid'])


# In[11]:


" ".join(sorted(list(allChars)))


# In[12]:


first_sentences_test = []
second_sentences_test = []
labels_test = []
guids = []
for x in test_data:
    first_sentences_test.append(x['sentence1']+x['sentence2'])
    second_sentences_test.append(x['sentence2'][:-1])
    labels_test.append(0)
    guids.append(x['guid'])


# In[13]:


#Vectorizing the data using word
vectorizer = TfidfVectorizer(lowercase=False,analyzer = 'word',ngram_range=(1,1),norm='l2',max_features=12000,preprocessor = custom_tokenizer)
vectorizer.fit(first_sentences_train)

#vectorizing the data using char
vectorizer2 = TfidfVectorizer(lowercase=False,analyzer = 'char',ngram_range=(1,5),norm='l2',max_features=12000,preprocessor=custom_tokenizer)
vectorizer2.fit(first_sentences_train)

#converting to dense array

tf_first_sentences_train=vectorizer.transform(first_sentences_train).toarray()
#tf_second_sentences_train=vectorizer.transform(second_sentences_train).toarray()

#converting to dense array
tf_first_sentences_train2=vectorizer2.transform(first_sentences_train).toarray()
#tf_second_sentences_train2=vectorizer2.transform(second_sentences_train).toarray()

#print(tf_first_sentences_train.shape)
#print(tf_second_sentences_train.shape)
#print(tf_first_sentences_train2.shape)
#print(tf_second_sentences_train2.shape)



# In[14]:


train_data = np.concatenate([tf_first_sentences_train,tf_first_sentences_train2],axis=1)
#Concatenating the vectors from the word and character vectorization
#train_data = np.array(tf_first_sentences_train)


# In[15]:


import gc 
#cleaning memory
tf_first_sentences_train = None 
tf_first_sentences_train2 = None
tf_second_sentences_train  = None
tf_second_sentences_train2 = None
gc.collect()


# In[16]:


# scaler = StandardScaler() 
#Tried using a scaler on the concatenated data, worse result

# scaler.fit(train_data)

# train_data = scaler.transform(train_data)


# In[ ]:





# In[17]:


#print(tf_first_sentences_train.shape)
#print(tf_second_sentences_train.shape)

#print(train_data.shape)


# In[18]:


train_data=TensorDataset(torch.tensor(train_data,dtype=torch.float32),torch.tensor(labels_train,dtype=torch.long)) 
#We create a dataset object
train_loader=DataLoader(train_data,batch_size=128,shuffle=True) 
#The dataloader will be used to loop through the data


# In[19]:


tf_first_sentences_valid=vectorizer.transform(first_sentences_valid).toarray()
#tf_second_sentences_valid=vectorizer.transform(second_sentences_valid).toarray()
tf_first_sentences_valid2=vectorizer2.transform(first_sentences_valid).toarray()
#tf_second_sentences_valid2=vectorizer2.transform(second_sentences_valid).toarray()

#print(tf_first_sentences_valid.shape)
#print(tf_second_sentences_valid.shape)
valid_data = np.concatenate([tf_first_sentences_valid,tf_first_sentences_valid2],axis=1)
#valid_data = np.array(tf_first_sentences_valid)
# valid_data = scaler.transform(valid_data)

valid_data=TensorDataset(torch.tensor(valid_data,dtype=torch.float32),torch.tensor(labels_valid,dtype=torch.long))
#same thing as for training
valid_loader=DataLoader(valid_data,batch_size=128,shuffle=True)


# In[ ]:





# In[20]:


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        #We declare the model
        self.lin1 = nn.Linear(24000,128) 
        #fully connected layers
        self.fc1 = nn.Linear(128,32)
        self.fc3 = nn.Linear(32,4)
        self.activation = nn.ReLU() 
        #activation function
        self.dropout = nn.Dropout(0.2) 
        # dropout
        self.fullyConnected = nn.Sequential(self.lin1,self.activation,self.dropout,self.fc1,self.activation,nn.Dropout(0.2),self.fc3)
        
    def forward(self, x):        
        
        output = self.fullyConnected(x)
        #print(output.shape)
        return output


# In[ ]:





# In[26]:


neuralClassifier=None
neuralClassifier = Model().cuda()
#We declare our model and pass it to cuda
neuralClassifier.train()


# In[ ]:





# In[ ]:





# In[27]:


allLabels = []

for i, x in tqdm(enumerate(train_loader, 0)):
    inp, lab = x
    #we get all the test labels
    allLabels+=list(lab.cpu().numpy())
    
allLabels = np.array(allLabels)

print(allLabels.shape)

fullWeights = []
for i in range(4): 
    #we compute the class weights being inversely proportional to the frequencies
    fullWeights.append(len(allLabels)/len(allLabels[allLabels==i]))
    


# In[28]:


full_weights = torch.FloatTensor([22.420524691358025,44.70307692307692,2.259311095560221,2.0390877192982457]).to('cuda')
#Class weights based on the formula above
full_weights = nn.functional.normalize(full_weights,p=1.0,dim=-1)
print(full_weights)
#We normalize the weights vector with the l1 norm

lossFct = nn.CrossEntropyLoss(weight = full_weights)

#choosing the loss function

gradientOptimizer = optim.Adam(neuralClassifier.parameters(), lr=0.001)

#We choose the adam optimizer


# In[ ]:





# In[40]:


for step in range(10):
    
    neuralClassifier.train()
    fullLoss=0
    
    allLabels = []
    allPredictions = []
    
    #storing all labels to compute metrics
    
    for _,current_training_batch in tqdm(enumerate(train_loader)):
        
        
        gradientOptimizer.zero_grad()
        
        
        samples,labels = current_training_batch    
        # Data from current batch 
        
        
        
        #Reset all gradients
        
        samples = samples.to('cuda')
        labels = labels.to('cuda')        
        
        #we pass all parameters to cuda
        
        weights = []             
        for k in range(4):
            if labels.eq(k).sum().item(): 
                weights.append(labels.shape[0]/labels.eq(k).sum().item())
            else:
                weights.append(0)
        weights = torch.tensor(weights,dtype=torch.float32)
        

        weights = nn.functional.normalize(weights,p=1.0,dim=-1)
        
        #Failed attempt to use the weights in the current batch. Failed in terms of not bringing any improvement
        
        if i==0:
            print(weights)
        
        outputs = neuralClassifier(samples)
        
        
        allLabels+=labels.cpu()
        allPredictions+=outputs.detach().cpu()
         
        currentLoss = lossFct(outputs, labels)
        
        #We compute the loss
        
        currentLoss.backward()
        
        #Backwards propagation using our loss function
        
        gradientOptimizer.step()
        
        #Gradient decent
        
        fullLoss+=currentLoss.item()
        
        #The sum of losses over all batches will be the training loss for this epoch
    
    #torch.save(neuralClassifier.state_dict(),f'./checkpoints/checkpoint_{step}')
    #We save a checkpoint of the model
    
    l=len(train_loader)   
    epoch_loss = fullLoss/l
    print(f"epoch {step}, train loss={epoch_loss}",end='\n')
    neuralClassifier.eval()
    
    #We get the model into eval mode
    allLabels = []
    allPredictions = []
    
    valid_loss=0
    for _,current_valid_batch in tqdm(enumerate(valid_loader)):
        #Used tqdm to see the training progress better
        samples,labels = current_valid_batch
        samples = samples.to('cuda')
        labels = labels.to('cuda')
        
        #Same procedure as with training
        outputs = neuralClassifier(samples)
        
        #We just compute the loss, but no backprop at this step. we only test
        loss2 = lossFct(outputs, labels)
        valid_loss+=loss2
        
        #Total validation loss
        
        allLabels+=labels.cpu()
        allPredictions+=outputs.detach().cpu()
        
        #Every prediction and label
    allPredictions = [torch.argmax(x).item() for x in allPredictions]
    print(f"f1-score for validation={f1_score(allLabels,allPredictions,average='macro')} and loss={valid_loss/len(valid_loader)}")


# In[30]:


print(f1_score(allLabels,allPredictions,average=None))

#Computing the f1 score for each class


# In[ ]:





# In[31]:


confusionMatrix = confusion_matrix(allLabels,allPredictions)
seaborn.heatmap(confusionMatrix,annot=True)
plt.show()

#Confusion matrix


# In[32]:


print(classification_report(allLabels,allPredictions))

#More metrics


# In[33]:


first_sentences_test = []
second_sentences_test = []
labels_test = []
guids = []
for x in test_data:
    first_sentences_test.append(x['sentence1']+x['sentence2'])
    #Building the test data
    second_sentences_test.append(x['sentence2'][:-1])
    print(x['sentence1'])
    labels_test.append(0)
    guids.append(x['guid'])


# In[34]:


tf_first_sentences_test=vectorizer.transform(first_sentences_test).toarray()
#tf_second_sentences_test=vectorizer.transform(second_sentences_test).toarray()
tf_first_sentences_test2=vectorizer2.transform(first_sentences_test).toarray()
#tf_second_sentences_test2=vectorizer2.transform(second_sentences_test).toarray()
#print(tf_first_sentences_test.shape)
#print(tf_second_sentences_test.shape)
test_data = np.concatenate([tf_first_sentences_test,tf_first_sentences_test2],axis=1)

#Vectorizing and concatenate test data

#test_data = np.array(tf_first_sentences_test)

# test_data = scaler.transform(test_data)

test_data = TensorDataset(torch.tensor(test_data,dtype=torch.float32),torch.tensor(labels_test,dtype=torch.long))
test_loader = DataLoader(test_data,batch_size=1,shuffle=False)


# In[35]:


neuralClassifier.eval()

predictions = []

with torch.no_grad():
    
    for i, current_test_batch in tqdm(enumerate(test_loader)):
        samples, labels = current_test_batch
        samples = samples.to('cuda')
        labels = labels.to('cuda')
        outputs = neuralClassifier(samples)
        #We run the data through the model
        outputs = nn.Softmax(dim=1)(outputs)
        #We use softmax on the predictions (used threshholds for classes, but was not successful)
        predictions+=[torch.argmax(outputs).item()]
        #The highest probability class is out prediction
        


# In[36]:


#for (i,x) in enumerate(second_sentences_test):


# In[37]:


with open("submission3.csv","w") as fout:
    
    print("guid,label",file=fout) #output the prediction to a csv
    for x,y in zip(guids,predictions):
        print(f"{x},{y}",file=fout)
        
    


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


#print(train_labs[0].item())


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




