import os, sys
import pandas as pd 
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
from datetime import datetime
import datetime as dt
import featexp
import sklearn
from scipy.stats import *
from math import sqrt
os.chdir("/Users/supark/Downloads")
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)

data = pd.read_csv('concha_data.csv')

# set other parameters as index 
data.set_index(['test_date','nid', 'gender', 'naics', 'age_group', 'region', 'NAICS_descr'], inplace=True)

# divide left and right to merge together 
left = data.iloc[:,0:7]
right = data.iloc[:,7:14]

left = left.reset_index()
right = right.reset_index()

# create a new column called "left" to distinguish left from right if needed
left['left'] = 1  
right['left'] = 0 

# rename columns to remove L/R
left = left.rename(columns={'L500k':'500k', 'L1k': '1k', 'L2k': '2k', 'L3k': '3k', 'L4k': '4k', 'L6k':'6k', 'L8k':'8k'})
right = right.rename(columns={'R500k':'500k', 'R1k': '1k', 'R2k': '2k', 'R3k': '3k', 'R4k': '4k', 'R6k':'6k', 'R8k':'8k'})

# concat and drop nas 
merge = pd.concat([left, right])
merge = merge.dropna() 

data = merge 
# set other parameters as index 
data.set_index(['test_date','nid', 'gender', 'naics', 'age_group', 'region', 'NAICS_descr'], inplace=True) 

# standardize the data attributes
data = pd.DataFrame(preprocessing.scale(data), columns = data.columns, index = data.index)

from sklearn.model_selection import train_test_split

# Separate features and targets
X = data[['2k', '4k', '6k']].to_numpy()
y = data[['500k', '1k', '3k', '8k']].to_numpy()

# split into train and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

# # neural network using Keras
# Load dependencies
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
import numpy as np

# Set the input shape
input_shape = (3,)
print(f'Feature shape: {input_shape}')

# Create the model
model = Sequential()
model.add(Dense(16, input_shape=input_shape, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='linear'))

# Configure the model and start training
model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_squared_error'])


# randomized search to tune hyperparameters
from sklearn.model_selection import * 
from keras.wrappers.scikit_learn import KerasRegressor

def create_model():
# create model
    model = Sequential()
    model.add(Dense(16, input_shape=input_shape, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(4, activation='linear'))
# Compile model
    model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_squared_error'])
    return model

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# create model
nn_keras = KerasRegressor(build_fn=create_model, verbose=0)

# define the grid search parameters
batch_size = [10, 20, 40, 60, 80, 100]
epochs = [10, 50, 100]
param_grid = dict(batch_size=batch_size, epochs=epochs)
rd = RandomizedSearchCV(nn_keras, param_distributions=param_grid, n_jobs=1, cv=5, scoring='neg_mean_absolute_error', verbose=1)
rd_result = rd.fit(X_train, y_train)


# summarize results
print("Best: %f using %s" % (rd_result.best_score_, rd_result.best_params_))
means = rd_result.cv_results_['mean_test_score']
stds = rd_result.cv_results_['std_test_score']
params = rd_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


# simple interface

twok = float(input('Enter 2k threshold value: '))
fourk = float(input('Enter 4k threshold value: '))
sixk = float(input('Enter 6k threshold value: '))
x_test = [[twok, fourk, sixk]]
print("500k, 1k, 3k, and 8k values: ", dt.predict(x_test))


# API using AWS

model.save('concha_nn')
from tensorflow import keras
nn = keras.models.load_model('concha_nn_0409')

