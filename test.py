__author__ = 'Ajay Arunachalam'
__version__ = '0.0.1'
__date__ = '1.9.2021'

# load libraries

import pandas as pd
import numpy as np
data = pd.read_csv('./dataset_Facebook.csv', sep=';')
print(data.shape)

data.head()

# load pymhopt library
from pyOP.optimization import *

# Read Data

data_fb, df_fb = Optimization.read_data(data)

print(data_fb.shape)

print(df_fb.shape)

# check missing values
def missing(x):
    return sum(x.isnull())


print("Checking for Missing row & col wise")
print('**************************')
print("Missing values per row:")
print(df_fb.apply(missing, axis=1))
print("Missing values per column:")
print(df_fb.apply(missing, axis=0))

# Select configurations

NUM_FEATURES, SELECTED_FEATURES, OPTIMIZATION_TARGET, CATEGORICAL_SELECT, NUM_GENERATIONS = Optimization.set_config(NUM_FEATURES=18, SELECTED_FEATURES=['Category', 'Type', 'Post Month', 'Post Weekday', 'Post Hour', 'Paid'], OPTIMIZATION_TARGET='Total Interactions',  CATEGORICAL_SELECT=['Category', 'Type', 'Paid'],
                                                                                                 NUM_GENERATIONS=20)

# Genetic Programming with symbolic regression 

Optimization.symbolic(df=data_fb, selected_features=SELECTED_FEATURES, target=OPTIMIZATION_TARGET, categorical_cols=CATEGORICAL_SELECT,generations=NUM_GENERATIONS)


# Symbolic regression with multi-objective genetic programming using NSGA-II 

Optimization.nsgaII(df=data_fb, selected_features=SELECTED_FEATURES, target=OPTIMIZATION_TARGET, categorical_cols=CATEGORICAL_SELECT, max_generations=20)