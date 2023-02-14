#!/usr/bin/env python
# coding: utf-8

# In[199]:


import pandas as pd
import numpy as np
from numpy import array
import PIL
import os
from PIL import Image
classes=[
    "bishop",
    "king",
    "knight",
    "pawn",
    "queen",
    "rook",
]
print(Image.open('C:\\Users\\ahmed\\Downloads\\Compressed\\archive_3\\data\\bishop\\100.png'))


# In[200]:


def load_img_vs_class():
    directorys={}
    for root, folders, files in os.walk(os.getcwd()):
         for filename in files:
            #directory.appen({filename})
            if filename.endswith(".png"):
                directorys[root+"\\"+filename]=root.split("\\")[7]
                #print(root+" Contain"+filename)
    return directorys


# In[201]:


data=load_img_vs_class()


# In[202]:


list(data.items())


# In[203]:


dataset=pd.DataFrame(data.items(),columns=["image","class"])


# In[204]:


dataset.to_csv("index.csv")


# In[205]:


def read_img(path):
    return array(Image.open(path))


# In[206]:


dataset["image_map"]=dataset["image"].map(read_img)


# In[207]:


print(dataset["image_map"].values[0].shape)


# In[208]:


def encode(value):
    try:
        return classes.index(value)
    except:
        return 0


# In[209]:


dataset["class"]=dataset["class"].map(encode)


# In[210]:


dataset["class"].unique()


# In[211]:


def convert_and_stander(img):
    tensor=[]
    for col in range(85):
        for row in range(85):
            tensor.append(float(img[col][row]))
    return np.array(tensor)


# In[212]:


dataset["image_map"]=dataset["image_map"].values


# In[ ]:





# In[215]:


dataset["image_map"].values[0]


# In[216]:


height=85
width=85


# In[218]:


x=dataset["image_map"]
y=dataset["class"]


# In[219]:


print(type(x[0]))


# In[220]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=1)


# In[230]:


from keras.models import Sequential
from keras.layers import Dense,Flatten,Dropout,Conv2D,MaxPooling2D
from keras.utils import *
import keras
model=Sequential()
model.add(Conv2D(64,kernel_size=(10,10),input_shape=(85,85,3),activation="relu",kernel_initializer="he_uniform"))
model.add(MaxPooling2D(pool_size=(10,10)))
model.add(Flatten())
model.add(Dense(100,activation="relu"))
model.add(Dense(len(dataset["class"].unique()),activation="sigmoid"))
model.compile(optimizer='SGD',loss="binary_crossentropy",
metrics=["accuracy"])
model.summary()


# In[224]:


model.fit(x,y,epochs=10)


# In[ ]:





# In[ ]:




