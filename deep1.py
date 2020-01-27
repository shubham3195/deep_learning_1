#!/usr/bin/env python
# coding: utf-8

# In[651]:


#software 1

# import sys
# filename = sys.argv[1]
ind=[]
l=[]
for i in range(101,1001):
    ind.append(i)
    if(i%3==0 and i%5==0):
        l.append("fizzbuzz")
        continue
    elif(i%3==0):
        l.append("fizz")
        continue
    elif(i%5==0):
        l.append("buzz")
        continue
    else:
        l.append(i)
#print(l)

f=open("/home/shubham/Desktop/Software1.txt","w")
for i in l:
    f.write(str(i)+"\n")


# In[652]:


#software2

import torch
import pandas as pd
data=pd.read_csv("/home/shubham/Desktop/input/input.txt", sep="\t")

df = pd.DataFrame(list(zip(ind,l)), columns = ['feature', 'label'])  

# from sklearn.preprocessing import LabelEncoder 
# le = LabelEncoder()  
# data['feature']= le.fit_transform(data['feature'])
# data['label']= le.fit_transform(data['label'])


# In[653]:


def decimalToBinary(n):  
    x=bin(n)[2:].zfill(10)
    a=list()
    for i in range(len(x)):
        a.append(int(x[i]))
    return a
# def decimalToBinary(n):    
#     if(n > 1):  
#         # divide with integral result  
#         # (discard remainder)  
#         decimalToBinary(n//2)     
#     #print(n%2, end=' ') 
def perform(l):
    a=list()
    for i in l:
        a.append(decimalToBinary(i))
    return a


# In[ ]:





# In[654]:


# def one_hot_encode(out):
#     if out == 'fizz':
#         return np.array([1, 0, 0, 0])
#     elif out == 'buzz':
#         return np.array([0, 1, 0, 0])
#     elif out == 'fizzbuzz':
#         return np.array([0, 0, 1, 0])
#     else:
#         return np.array([0, 0, 0, 1])

def one_hot_encode(out):
    if out == 'fizz':
        return np.array(0)    # div by 3
    elif out == 'buzz':
        return np.array(1)# div by 5
    elif out == 'fizzbuzz':
        return np.array(2)# div by 3 and 5 both
    else:
        return np.array(3)


def enu(l):
    a=list()
    for i in l:
        a.append(one_hot_encode(i))
    return np.array(a)


# In[655]:


import numpy as np
#from torchvision import transforms


# In[656]:


trx=np.arange(101,1001)
tx=torch.FloatTensor(perform(trx)).unsqueeze(dim=1)


ty=torch.LongTensor(enu(l))
# train_x_raw = np.arange(train_x_start, train_x_end + 1)
# train_x = binary_encode_16b_array(train_x_raw).reshape([-1, 16])
# train_y_raw = fizzbuzz(train_x_start, train_x_end)
# train_y = one_hot_encode_array(train_y_raw)


# In[657]:


# Neural Network
#print(tx[0:20])
print(ty[0:20])


# In[658]:


from torch import nn
import torch.nn.functional as F
# from torchvision import datasets, transforms
#f1=open("/home/shubham/Desktop/Software2.txt","w")


# In[659]:


model = nn.Sequential(nn.Linear(10, 128),
                      nn.ReLU(),
                      nn.Linear(128, 128),
                      nn.ReLU(),
                      nn.Linear(128, 128),
                      nn.ReLU(),
                      nn.Linear(128, 4))


# In[660]:


#criterion = nn.MSELoss()
criterion = nn.CrossEntropyLoss()# Optimizers require the parameters to optimize and a learning rate
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)


# In[661]:


epochs = 1024
for e in range(epochs):
    running_loss = 0
    for i in range(0,len(tx),64):
        # Flatten MNIST images into a 784 long vector
        t=tx[i:i+64].view(tx[i:i+64].shape[0], -1)
        #t = torch.from_numpy(tx[i])
        #print(t.shape)
        # Training pass
        optimizer.zero_grad()
        
        output = model(t)
        
        #print(output)
        #print(ty[i:i+32])
        #output[0]=transforms.Normalize(2, 0.5)(output)
        #print("cytcjbiuh",output.shape,ty[i:i+32].shape)
        #print("cytcjbiuh",output.squeeze(),ty[i:i+32].view(-1))
        
        loss = criterion(output.squeeze(), ty[i:i+64].view(-1))
        
        #print("loss :",loss)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        #print(running_loss)
    else:
        print(f"Training loss: {running_loss/len(tx)}")
        


# In[ ]:





# In[662]:


torch.save(model, "/home/shubham/Desktop/model.pt")


# In[663]:



m= torch.load("/home/shubham/Desktop/model.pt")
m.eval()


# In[ ]:




