#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
import pickle
from flask import Flask, request


# In[ ]:


app=Flask(__name__)
pickle_in=open("classifier.pkl", "rb")
classifier=pickle.load(pickle_in)

@app.route('/')
def welcome():
    return "welcome you"

@app.route("/predict")
def predict_note_authentication():
    variance=request.args.get('variance')
    skewness=request.args.get('skewness')
    curtosis=request.args.get('curtosis')
    entropy=request.args.get('entropy')
    prediction=classifier.predict([[variance,skewness,curtosis,entropy]])
    return "the prediction values is" + str(prediction)

@app.route("/predict_file", methods=['POST'])
def predict_note_file():
    df_test=pd.read_csv(request.files.get("file"))
    prediction=classifier.predict(df_test)
    return "the prediction list for the csv file" + str(list(prediction))

if __name__=='__main__':
    app.run()


# In[ ]:




