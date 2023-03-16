#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 15:44:46 2023

@author: mfeene
# Run in Google colab for GPU
"""


# -------------------------
#  3. Creating the Model
# -------------------------

# Run first for reproducability
from numpy.random import seed
seed(1)
from tensorflow import random
random.set_seed(1)

# !pip install keras_tuner
# !pip install tensorflow-addons
# !pip install scikeras

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import tensorflow as tf
import tensorflow_addons as tfa
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import Adam, SGD
import keras_tuner as kt
from keras_tuner import BayesianOptimization
from keras.callbacks import EarlyStopping


# -------------------------
# Data Loading 
# -------------------------
# Read in data
final_model_data = pd.read_csv('/content/final_model_data_addl.csv')

# Split into X and Y
X = final_model_data.iloc[:, 4:-1].values
y = final_model_data.iloc[:, -1].values

# Create a train-test split with ratio 30:70
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size = 0.30, random_state = 123)

# Scale the training and testing data
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Save the scaler for future use
scalerfile = 'scaler.save'
pickle.dump(sc, open(scalerfile, 'wb'))

# State evaluation metrics
threshold = 0.5
f1_macro = tfa.metrics.F1Score(num_classes = 1, threshold = threshold, average = 'macro') # macro weighted F1 score
precision = tf.keras.metrics.Precision(thresholds = threshold)
recall = tf.keras.metrics.Recall(thresholds = threshold)
roc_auc = tf.keras.metrics.AUC(curve = 'ROC') # roc auc
earlyStopping = EarlyStopping(monitor = 'val_accuracy', patience = 5) # early stopping to optimize training


# -------------------------
# Keras Tuner
# -------------------------
hidden_layer_act = tf.keras.layers.LeakyReLU(alpha = 0.01)

def build_model(hp):

    model = Sequential()

    model.add(Dense(500, input_dim = X_train.shape[1], activation = hidden_layer_act))
   
   # Dense layers
    for i in range(hp.Int('n_layers_dense', 1, 3)):
        model.add(Dense(hp.Int('dense_units', min_value = 10, max_value = 100, step = 10), activation = hidden_layer_act))
 
    # Batch normalization
    if hp.Boolean('batch_norm'):
        model.add(BatchNormalization())
    
    # Dense layers
    for i in range(hp.Int('n_layers_dense2', 1, 3)):
        model.add(Dense(hp.Int('dense_units2', min_value = 10, max_value = 100, step = 10), activation = hidden_layer_act))
    
    # Batch normalization
    if hp.Boolean('batch_norm2'):
        model.add(BatchNormalization()) 

    # Dense layers
    for i in range(hp.Int('n_layers_dense3', 1, 3)):
        model.add(Dense(hp.Int('dense_units3', min_value = 10, max_value = 50, step = 10), activation = hidden_layer_act))
 
    model.add(Dense(1, activation = 'sigmoid'))

    opt = hp.Choice('optimizer', values = ['adam'])
                        
    model.compile(loss = 'binary_crossentropy', optimizer = opt, metrics = ['accuracy', f1_macro, precision, recall, roc_auc])
    return model

build_model(kt.HyperParameters())


# Search for the best hyperparameters
mm_tuner = kt.BayesianOptimization(build_model,
                     objective = kt.Objective('val_accuracy', direction = 'max'),
                     seed = 123,
                     max_trials = 20,
                     directory = 'mm_model_trial')

mm_tuner.search(X_train, y_train, epochs = 20, validation_split = 0.2, callbacks = [earlyStopping]) 


# Get the optimal hyperparameters
mm_best_hps = mm_tuner.get_best_hyperparameters(num_trials = 1)[0]
mm_best_hps


# Build the model with the optimal hyperparameters and train it on the data
mm_model = mm_tuner.hypermodel.build(mm_best_hps)
mm_history = mm_model.fit(X_train, y_train, epochs = 100, validation_split = 0.2, verbose = 2)


# Identify best epoch
mm_val_acc_per_epoch = mm_history.history['val_accuracy']
mm_best_epoch = mm_val_acc_per_epoch.index(max(mm_val_acc_per_epoch)) + 1
print('Best epoch: %d' % (mm_best_epoch,))


# Re-instantiate hypermodel and train on optimal epoch count from above
mm_hypermodel = mm_tuner.hypermodel.build(mm_best_hps)


# Retrain 
mm_hypermodel.fit(X_train, y_train, epochs = mm_best_epoch, validation_split = 0.2, verbose = 2)
mm_hypermodel.save_weights('mm_model_weights.h5') # save the weights of the optimally performing model


# Evaluate hypermodel with the optimal parameters on the test set data
mm_hypermodel.load_weights('mm_model_weights.h5')
mm_test_result = mm_hypermodel.evaluate(X_test, y_test, verbose = 1)


# Capture metrics
mm_test_loss = mm_test_result[0] # loss
mm_test_accuracy = mm_test_result[1] # accuracy
mm_test_f1_score = mm_test_result[2] # macro f1 score
mm_test_precision = mm_test_result[3] # precision
mm_test_recall = mm_test_result[4] # recall
mm_test_roc_auc = mm_test_result[5] # roc auc


# Display metrics
metric_names = pd.DataFrame(['loss', 'accuracy', 'macro f1 score', 'precision', 'recall', 'roc auc'])
mm_test_result_df = pd.DataFrame([mm_test_loss, mm_test_accuracy, mm_test_f1_score, mm_test_precision, mm_test_recall, mm_test_roc_auc])
mm_result = pd.concat([metric_names, mm_test_result_df], axis = 1)
mm_result['Model Type'] = '1. March Madness Model'
mm_result.columns = ['Test Set Metric', 'Test Set Metric Value', 'Model Type']
mm_metrics = mm_result[['Model Type', 'Test Set Metric', 'Test Set Metric Value']]
mm_metrics

# Architecture summary
mm_hypermodel.summary()


# -------------------------
# Evaluating the Model on Test data
# -------------------------
preds = mm_hypermodel.predict(X_test)
preds_clean = np.where(preds >= threshold, 1, 0)

# Accuracy
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, preds_clean))

# ROC AUC
from sklearn.metrics import roc_auc_score
print(roc_auc_score(y_test, preds_clean))

# Classification Report
from sklearn.metrics import classification_report
print(classification_report(y_test, preds_clean))

# Save model as pkl file
with open('mm_hypermodel.pkl', 'wb') as fid:
    pickle.dump(mm_hypermodel, fid)
    
