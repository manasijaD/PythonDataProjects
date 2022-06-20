#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
df = pd.read_csv('house_price_data.csv')
df
dataset = df.values
dataset
X = dataset[:,0:10]
Y = dataset[:,10]


# In[2]:


#Data pre-processing
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
X_scale = min_max_scaler.fit_transform(X)
X_scale


# In[3]:


from sklearn.model_selection import train_test_split
#Data splitting
X_train, X_val_and_test, Y_train, Y_val_and_test = train_test_split(X_scale, Y, test_size=0.3)
X_val, X_test, Y_val, Y_test = train_test_split(X_val_and_test, Y_val_and_test, test_size=0.5)
print(X_train.shape, X_val.shape, X_test.shape, Y_train.shape, Y_val.shape, Y_test.shape)


# In[8]:


#Model_2
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense 
model_2 = Sequential([Dense(1000, activation='relu', input_shape=(10,)),
Dense(1000, activation='relu'),
Dense(1000, activation='relu'),
Dense(1000, activation='relu'),
Dense(1, activation='sigmoid'),])
model_2.compile(optimizer='adam',
loss='binary_crossentropy',
metrics=['accuracy'])
hist_2 = model_2.fit(X_train, Y_train,batch_size=32,epochs=100,validation_data=(X_val, Y_val))


# In[10]:


#Testing Model_2
model_2.evaluate(X_test, Y_test)


# In[14]:


import matplotlib.pyplot as plt 
plt.plot(hist_2.history['loss'])
plt.plot(hist_2.history['val_loss'])
plt.title('Model_2 loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.show()


# In[15]:


plt.plot(hist_2.history['accuracy'])
plt.plot(hist_2.history['val_accuracy'])
plt.title('Model_2 accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='lower right')


# In[13]:


from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(X_test, Y_test)
reg.score(X_test, Y_test)

