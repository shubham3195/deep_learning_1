#!/usr/bin/env python
# coding: utf-8

# In[283]:


# main.py to generate software1 and software2
#preprocessing train data
import pandas as pd
import numpy as np
import torch
data=pd.read_csv("test_input.txt", sep="\n", header=None)
#print(len(data))

ind=[]
l=[]
for i in range(1,len(data)+1):
    ind.append(int(i))
    i=int(i)
    
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

f=open("Software1.txt","w")
for i in l:
    f.write(str(i)+"\n")


# In[303]:


#preprocess test for software2
print("Department of Computer Science & Automation")
print("Name: Shubham Sharma")
print("Mtech,16013")
print(" ")
print("Software1.txt created")
def decimalToBinary(n):  
    x=bin(n)[2:].zfill(10)
    a=list()
    for i in range(len(x)):
        a.append(int(x[i]))
    return a

def perform(l):
    a=list()
    for i in l:
        a.append(decimalToBinary(i))
    return a

trx=np.array(ind)
tx=torch.FloatTensor(perform(trx)).unsqueeze(dim=1)


# In[285]:


import torch
m= torch.load("model/model.pt")
m.eval()


# In[286]:


f1=open("Software2.txt","w")
infer=list()
for i in range(len(tx)):
    output=m(tx[i])
    #print(output)
    
    lp=output.max(1)[1]
    max_index=lp
    #print(lp)
#     label=output.max()
    
#     print("label:",label)
    
#     li=list(output)
#     ma=-999999999
#     for i in range(len(li)):
#         if(li[i]>ma):
#             ma=li[i]
#     print("ma",ma)
#     max_value = max(li)
#     maxpos = li.index(max(li))
    
#     print(maxpos)
    
    
    if(max_index==0):
        f1.write("fizz\n")
        infer.append("fizz")
    elif(max_index==1):
        f1.write("buzz\n")
        infer.append("buzz")
    elif(max_index==2):
        f1.write("fizzbuzz\n")
        infer.append("fizzbuzz")
    else:
        f1.write(str(i+1)+"\n")
        infer.append(str(i+1))
        
    #f.write(str(i)+"\n")


# In[297]:


print("Software2.txt created")
# Accuracy
#for fizz
c=0
o=0
for i in range(len(l)):
    if(l[i]=='fizzbuzz'):
        o=o+1
        if(infer[i]=='fizzbuzz'):
            c=c+1


# In[299]:


#c


# In[289]:


#print(l[2],infer[2])
#print(l[1],infer[1])


# In[300]:


c1=0
ov=0
for i in range(len(l)):
    if(l[i]!='fizz' and l[i]!='buzz' and l[i]!='fizzbuzz'):
        #print(l[i],infer[i])
        if(l[i] == infer[i]):
            c1=c1+1
        ov=ov+1


# In[291]:


#c1


# In[ ]:





# In[ ]:




