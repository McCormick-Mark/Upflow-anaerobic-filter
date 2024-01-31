# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 11:36:35 2020

@author: mark_
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 20:10:17 2020

@author: mark_
"""

#### This file contains:
    # A function to build, train, evaluate and save the MLP model
    # A function to build, train, evaluate and save the 1-layer perceptron model
    # A function to load the NN models and use them for simulation

import numpy as np
# np.require("np==1.19.2")
# import np
import pandas as pd
import tensorflow# as tf 
import datetime
import random
import csv
import scipy
import itertools
# import os


import sklearn
from sklearn.model_selection import train_test_split
from sklearn.decomposition import FastICA, PCA
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn import linear_model#, LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from math import sqrt
from scipy import stats

# Don't import statsmodels when running on HPC clusters
# import statsmodels
# from statsmodels.stats.multitest import multipletests

import tensorflow as tf    # copied from Keras site
from tensorflow import keras # copied from Keras site
from tensorflow.keras import layers  # copied from Keras site

import tensorflow.keras
from tensorflow import keras
#from keras import layers  Do not import this because ...
from tensorflow.keras import activations
# from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout, Activation, LSTM #, Conv1D, Flatten, Reshape, LSTM, MaxPooling1D, GlobalAveragePooling1D, concatenate, Concatenate
from tensorflow.keras.optimizers import RMSprop, SGD, Adam
from tensorflow.keras.initializers import Constant, RandomUniform, glorot_normal, glorot_uniform
# from keras import losses
# from keras import metrics
# from keras import initializers
from tensorflow.keras.losses import mse, mae, mape, binary_crossentropy, sparse_categorical_crossentropy, mean_squared_logarithmic_error
from tensorflow.keras.metrics import mse, mae#, mape	 the loss function is used when training the model # the metrics are not used when training the model. Any loss function may also be used as a metric function.
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model, to_categorical
# from keras.layers.merge import concatenate
from tensorflow.keras.models import model_from_json, load_model
from tensorflow.keras.regularizers import l1, l2


######### The MLP NN model   ##########################

# def AFBR_tensors(X_data, y_data, SX_test):

# ###Split data to obtain training and test data 
#     X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.33)  #, random_state=42
#     print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
      
#     plt.plot(X_train, marker='.', linestyle='None', c='g')
#         #plt.legend(labels, bbox_to_anchor=(1, 1)) #loc=9,
#     plt.title('Raw training data, X')
#     plt.xlabel('days')
#     # plt.show()
#     plt.savefig('C:/Users/mark_/Anaconda3/envs/tensorflow_env/mark/satw/AFBR_rawtraining.png', bbox_inches='tight')   
   
    
#     plt.plot(y_train, label = 'Biogas production [l/day]' )
#     plt.legend()
#     plt.title('Raw training data, target')
#     plt.xlabel('days')
#     #plt.show()
    
#     plt.plot(X_test, marker='.', linestyle='None', c='r')
#     #plt.legend(labels, bbox_to_anchor=(1, 1)) #loc=9,
#     plt.title('Raw test data, X')
#     plt.xlabel('days')
#     # plt.show()
    
#     plt.plot(y_test, label = 'Biogas production [l/day]')
#     plt.legend()
#     plt.title('Raw test data, target')
#     plt.xlabel('days')
#     # plt.show()

#     plt.plot(SX_test, label = 'Biogas production [l/day]')
#     #SX_test.plot()
#     plt.legend()
#     plt.title('Simulation data, target')
#     plt.xlabel('days')
#     # plt.show()
    
# ## Normalize training and testing data to a value between 0 and 1
#     scaler=preprocessing.MinMaxScaler(feature_range = (0,1), copy=True)
    
#     #scaler.fit(X_train)
#     X_train_mm = scaler.fit_transform(X_train)#, feature_range=(0,1)
#     plt.figure(figsize=(10,1.5))
#     plt.plot(X_train_mm)
#      #plt.legend(labels, bbox_to_anchor=(1, 1)) #loc=9,
#     plt.title('Scaled training data, X', fontsize='large')
#     plt.xlabel('Days')
#             #plt.savefig('Anaconda3/envs/tensorflow_env/mark/StrainX_365.svg', dpi=None, facecolor='w', edgecolor='w')
#     y_train= y_train.reshape(-1, 1)
#     #scaler.fit(y_train)
#     y_train_mm = scaler.fit_transform(y_train)#, feature_range=(0,1)
#     plt.figure(figsize=(10,1.5))
#     plt.plot(y_train_mm)
#       #plt.legend(labels, bbox_to_anchor=(1, 1)) #loc=9,
#     plt.title('Scaled training data, target', fontsize='large')
#     plt.xlabel('Days')       
    
#     X_test_mm = scaler.fit_transform(X_test)#, feature_range=(0,1)
#     plt.figure(figsize=(10,1.5))
#     plt.plot(X_test_mm)
#      #plt.legend(labels, bbox_to_anchor=(1, 1)) #loc=9,
#     plt.title('Scaled test data, X', fontsize='large')
#     plt.xlabel('Days')
#             #plt.savefig('Anaconda3/envs/tensorflow_env/mark/StrainX_365.svg', dpi=None, facecolor='w', edgecolor='w')
#     y_test= y_test.reshape(-1, 1)
#     y_test_mm = scaler.fit_transform(y_test)#, feature_range=(0,1)
#     plt.figure(figsize=(10,1.5))
#     plt.plot(y_test_mm)
#       #plt.legend(labels, bbox_to_anchor=(1, 1)) #loc=9,
#     plt.title('Scaled test data, target', fontsize='large')
#     plt.xlabel('Days')   
#     print('X_train', X_train.shape, 'y_train', y_train.shape, 'X_test', X_test.shape, 'y_test', y_test.shape)
    
# ## Normalize simulation testing data to a value between 0 and 1    
#     SX_test_mm = scaler.fit_transform(SX_test)#, feature_range=(0,1)
#     plt.figure(figsize=(10,1.5))
#     plt.plot(SX_test_mm)
#      #plt.legend(labels, bbox_to_anchor=(1, 1)) #loc=9,
#     plt.title('Scaled simulation data, X', fontsize='large')
#     plt.xlabel('Days')
#     print('SX_test_mm', SX_test_mm.shape)

#     plt.figure(figsize=(10,1.5))
#     plt.plot(SX_test_mm[:,3])
#      #plt.legend(labels, bbox_to_anchor=(1, 1)) #loc=9,
#     plt.title('Scaled simulation data, Influent flow rate', fontsize='large')
#     plt.xlabel('Days')
#     print('SX_test_mm[:,3] Influent flow rate', SX_test_mm.shape)
    
#     return X_train_mm, y_train_mm, X_test_mm, y_test_mm, SX_test_mm

#########    USE WHEN RUNNING GA ON THE CLUSTERS    ##########################
# archopti: Use to select the optimal number of hidden layers and hidden layer nodes

def AFBR_MLP_archopti(X_train_mm, y_train_mm, X_test_mm, y_test_mm, layer, node):  # , NN_arch, do  inputs, outputs, inputs_test, outputs_test    X_train_mm, y_train_mm, X_test_mm, y_test_mm
   
    X_train_mm = X_train_mm.astype(np.float32)  # This works

    y_train_mm = y_train_mm.astype(np.float32)# This works

    X_test_mm = X_test_mm.astype(np.float32)# This works

    y_test_mm = y_test_mm.astype(np.float32)# This works

    # layers = [2,3,4,5]#NN_arch[]
    node= node# [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    i = 0 # Initialize the number of hidden layers before each NN model build. Start with h = 1 => 2 layers 
    b = 4 #bch#4  #batch_size = 4 works best
    #[r[1] for r in NN_arch] # select the number of nodes
    lr = 0.0001  # 0.00001 works best with RMSProp. 0.000001 works best with Adam
    ep = 100# 200#400  # V1 best = 250
    #b1 = 0.9 # Used in Adam optimizer. Exponential decay rate for the first moment estimates. Default = 0.9
    do = 0.05 #0.05 # drop out rate
    #elu = keras.layers.ELU(alpha=0.9)
    rho = 0.99 #Discounting factor for the history/coming gradient. Defaults to 0.9.
    mom = 0.05#0.0 #Defaults to 0.0.
    
    initializer = tensorflow.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1, seed = None) #, seed=42, glorot_uniform(seed=42)  RandomNormal(mean=0.0, stddev=sdv, seed=42)
    # tf.keras.activations.relu(x, alpha=0.0, max_value=None, threshold=0)
   
    # sess.graph contains the graph definition; that enables the Graph Visualizer.

    # file_writer = tf.summary.FileWriter('\\Anaconda3\\envs\\Tensorflow_v2\\Mark\\', sess.graph)
    
    # with tf.Session() as sess:
    #     writer = tf.summary.FileWriter("\\Anaconda3\\envs\\Tensorflow_v2\\Mark\\", sess.graph)
    ######## SEQUENTIAL MODEL ###############
    #### Multi-layer. Comment when running 1-layer
####    # Build a new sequential model and run for each combination of number of hidden layers (2, 3, 4, 5 or 6) and units 
##    AFBR_opt = tensorflow.keras.Sequential()  # keras.
##    AFBR_opt.add(tensorflow.keras.layers.InputLayer(input_shape=(4,)))  #   keras.layers.Dense(4, input_shape=(4,)) keras.layers.InputLayer.  This works: AFBR.add(tensorflow.keras.Input(shape=(4,)))
##    AFBR_opt.add(tensorflow.keras.layers.Dense(units=node, activation= 'relu', use_bias=True, kernel_initializer=initializer, bias_initializer = 'zeros'))#
##    for i in range(layer):
##        AFBR_opt.add(Dropout(do))
##        AFBR_opt.add(tensorflow.keras.layers.Dense(units=node, activation= 'relu', use_bias=True, kernel_initializer=initializer, bias_initializer = 'zeros'))#, kernel_regularizer=l1(0.01), kernel_regularizer=l1(0.01) , activity_regularizer=l2(0.01)))#, kernel_regularizer='l1', activity_regularizer="l1"))
##    # # #     # i+=1
##    AFBR_opt.add(tensorflow.keras.layers.Dense(units=1)) #,, activation='linear',input_dim=(1),input_shape=(step,)  , input_shape=(24,8)units=2, , 
##    print(AFBR_opt.summary())
##    metric= tensorflow.keras.metrics.MeanSquaredError(name='mean_squared_error')
##    # metric= tensorflow.keras.metrics.MeanAbsoluteError(name='mean_absolute_error')
##    
##    AFBR_opt.compile(optimizer=tensorflow.keras.optimizers.RMSprop(learning_rate=lr, rho=rho, momentum = mom, epsilon=0.0000001, centered= True), loss=tensorflow.keras.losses.MeanSquaredError(),  metrics = [metric]) #,"mse" clipnorm=1.0 default: RMSprop(lr=0.001, rho=0.9, momentum=0.0, epsilon = 0.0000001, centered = False gives slightly better rmse) 
##    #Centere
##
##
##    history = AFBR_opt.fit(X_train_mm, y_train_mm, validation_data = (X_test_mm, y_test_mm), batch_size= int(b), epochs=ep,  shuffle = False, verbose=0)  # , inputs_test, outputs_test,  shuffle = True to shuffle before splitting., verbose=3, callbacks=[tensorboard_callback], tensorboard_callback, ICANN paper: 250 epochs    
##    # history_t = AFBR_t.fit(X_train_mm, y_train_mm, validation_data = (X_test_mm, y_test_mm), epochs=100, batch_size= int(b), shuffle = True, verbose=1) 
##    # history = AFBR.fit(X_train, y_train, epochs=100, batch_size=int(step), verbose=1)
##    test_scores = AFBR_opt.evaluate(X_test_mm, y_test_mm , verbose=0) #X_test_mm, y_test_mm
##    # print("Training loss:", history.history['loss'][0])
##    # print("Test loss:", history.history['val_loss'][1])
##    # print("Mean squared error:", test_scores[0])
##
##    # mse_train = test_scores[0]
##    # print("mse_train is: ", mse_train)
##
##    mse_eval = test_scores[0]
##    print("mse_eval (test) is: ", mse_eval)
##    # print("mse_eval is: ", + str(mse_eval.shape))
##    # print('input shape:%.1f%%', inputs)
##    # print('input test shape:%.1f%%', inputs_test)
##    
##    # prediction from training data
##    Yhat_o = AFBR_opt.predict([X_train_mm]).reshape(-1,1)
##    # np.savetxt("C:/Users/mark_/Documents/02_Professional/03_Unil/SATW/MDPI_paper/MDPI_article/MDPI_article_v03/figures/Yhat_train_mlp_noshuffle", Yhat, delimiter=',')
##    # np.savetxt('/scratch/mmccormi1/Yhat.csv', Yhat, delimiter=',')
##    print("Yhat shape is: " + str(Yhat_o.shape))
##    # prediction from validation test data
##    Yhat_v_o = AFBR_opt.predict([X_test_mm]).reshape(-1,1)#Y
##    print("Yhat_v shape is: " + str(Yhat_v_o.shape))
##    test_mse = sklearn.metrics.mean_squared_error(y_test_mm, Yhat_v_o, sample_weight=None, multioutput='uniform_average', squared=True)

    #########  1-layer model. If used, then add Yhat_v_mlp_1_layer to returned values
    Yhat_o = np.array(1)
    Yhat_v_o= np.array(1)
    test_mse = np.array(1)
    mse_eval = np.array(1)
    
    AFBR_1_layer = tensorflow.keras.Sequential()  # keras.
    AFBR_1_layer.add(tensorflow.keras.layers.InputLayer(input_shape=(4,)))
    AFBR_1_layer.add(tensorflow.keras.layers.Dense(units=node, activation= 'relu', use_bias=True, kernel_initializer=initializer, bias_initializer = 'zeros'))#
##    for i in range(layer):
##        AFBR_1_layer.add(Dropout(do))
##        AFBR_1_layer.add(tensorflow.keras.layers.Dense(units=node, activation= 'relu', use_bias=True, kernel_initializer=initializer, bias_initializer = 'zeros'))#, kern
    AFBR_1_layer.add(tensorflow.keras.layers.Dense(units=1)) #,, activation='linear',input_dim=(1),input_shape=(step,)  , input_shape=(24,8)units=2, , 
    print(AFBR_1_layer.summary())

    metric= tensorflow.keras.metrics.MeanAbsoluteError(name='mean_absolute_error')
    AFBR_1_layer.compile(optimizer=tensorflow.keras.optimizers.RMSprop(learning_rate=lr, rho=rho, momentum = mom, epsilon=0.0000001, centered= True), loss=tensorflow.keras.losses.MeanSquaredError(),  metrics = [metric]) #,"mse" clipnorm=1.0 default: RMSprop(lr=0.001, rho=0.9, momentum=0.0, epsilon = 0.0000001, centered = False gives slightly better rmse) 
    history = AFBR_1_layer.fit(X_train_mm, y_train_mm, validation_data = (X_test_mm, y_test_mm), batch_size= int(b), epochs=ep,  shuffle = False, verbose=0)
    Yhat_v_mlp_1_layer = AFBR_1_layer.predict([X_test_mm]).reshape(-1,1)

    
    ########    MDPI figure 4    ###########
    # MLP loss
##    plt.figure(figsize=(8,4))
##    # plt.title('MLP loss', fontsize='large') #and testing
##    plt.plot(history.history['loss'], c="k", linestyle="-", label='training_loss') #, 'r'train_history
##    plt.plot(history.history['val_loss'], c="b", linestyle="dotted", label='test_loss') #, 'b'val_history, val_loss
##    plt.legend()
##    plt.ylim(0.0, 0.015)  #0.003, 0.010
##    plt.xlabel('Epoch')
##    plt.ylabel('Mean Squared Error') #Mean squared 
##    plt.show()
    # plt.savefig('C:/Users/mark_/Documents/02_Professional/03_Unil/SATW/MDPI_paper/MDPI_article/MDPI_article_v12/figures/F7_mlp_loss.svg', format="svg")
    # plt.savefig('/scratch/mmccormi1/mlp_archopti_training.svg', format="svg")

    return Yhat_o, Yhat_v_o, test_mse, mse_eval, Yhat_v_mlp_1_layer   #  o = optimisation runs,  Yhat_v_mlp_1_layer, 

    # plt.scatter(y_test_mm, Yhat_v_o)



    
################ Replicated polynomial model building  ######################
def poly_rep(X_train_mm, y_train_mm, X_test_mm, y_test_mm, nrep):
# #### LOO replicates   #######
# ## Randomly select one set (line) to leave out. Repeat 100 times to make 27 separate datasets with one line left out
# ## 4 coefficients automatically generated because 4 is the number of features in the X_train_mm dataset
    import statsmodels
    from statsmodels.stats.multitest import multipletests
# # print('Replicated polynomial regression of surrogate AFBR data')
    y = y_train_mm #np.asarray(y_train_mm).reshape(-1,1) #np.asarray(y_train_mm)#.reshape(-1,1)  #SATWregdata.iloc[:,[0]] #SATWregdata.columns[0]
    #print('y before delete',y)
    X = X_train_mm #np.asarray(X_train_mm).reshape(-1,4) #np.asarray(X_train_mm) #SATWregdata.iloc[:,[1,2,5]]   # I_Q, I_Ti, I_PCS
##    print('X before delete',X)
    print('X_test_mm type in poly_rep', type(X_test_mm))
##    print('X_test_mm df', X_test_mm)
   
##    print('X_test_mm to_numpy', X_test_mm.to_numpy())
##    print('X_test_mm type numpy?', type(X_test_mm))
    # y_test_mm = np.asarray(y_test_mm) 
    # y_test_mm_ns = y_test_mm
    
    # X_test_mm = np.asarray(X_test_mm)
    
    Wr = []
    Ir = []
    Wr_all = []
    Ir_all = []
    y_test_loo_all = []
    yhat_poly = []
    yhat_reg = []
    yhat_regression = []#np.array(1)
    Yhat_polynomial =[]# np.zeros(len(y_test_mm))#[]
    Yhat_polynomial_all = []
    rms_poly = []
    result = []
    inter = []
    res = []
    Rsq = []
    Poly_sl = []
    Poly_pval=[]
    Poly_rmse = []
    Poly_reg = []
    Poly_r2_score=[]
    nrep = nrep#100
##    y_loo = np.arange(10)

##    rdel = np.random.randint(0, high=len(y))# rdel = random selection of the row to be deleted +1?
##    print('rdel',rdel)
##    y_loo = np.delete(y, rdel, axis=0)  #, axis=0
##    print('y_loo',y_loo)
##    X_loo = np.delete(X, rdel, axis=0) #np.asarray(X)
##    mdlr = linear_model.LinearRegression().fit(X_loo,y_loo)
##    Wr = np.asarray(mdlr.coef_)
##    print('Wr[0][0]',Wr[0][0])#[0][0])
##    Ir = mdlr.intercept_
##    print('X_test_mm[5,0]',X_test_mm[5,0])
##    print('X_test_mm type',type(X_test_mm[5,0]))
##    print('Wr[0][0] type',type(Wr[0][0]))
##
##    yhat_poly = float(X_test_mm[5,0])*Wr[0][0]
##    print('yhat_poly',yhat_poly)

    for j in range(nrep):
        y = y_train_mm
##        X = X_train_mm
        rdel = np.random.randint(0, high=len(y))# rdel = random selection of the row to be deleted +1?
##        print('rdel',rdel)
        y_loo = np.delete(y, rdel, axis=0)  #, axis=0
        #y_test_loo = np.delete(y_test_mm, rdel, axis=0) 
##        print('y_loo',y_loo)
        X_loo = np.delete(X, rdel, axis=0) #np.asarray(X)
        mdlr = linear_model.LinearRegression().fit(X_loo,y_loo)
        Wr = np.asarray(mdlr.coef_)
        #print('Wr',Wr)#[0][0])
        Ir = mdlr.intercept_
##        print('Ir[0]',Ir[0])
        #print('X_test_mm[5,0]',X_test_mm[5,0])
        #print('shape y_test_mm',y_test_mm.shape)
        for k in range(len(y_test_mm)):
##            yhat_poly.append((float(X_test_mm[k,0])*Wr[0][0] + float(X_test_mm[k,1])*Wr[0][1] + float(X_test_mm[k,2])*Wr[0][2] + float(X_test_mm[k,3])*Wr[0][3]+Ir)[0])  #Use [0] to select the array!  predicted response for each day of test data using the polynomial
            #print('yhat_poly',yhat_poly)
            yhat_poly = X_test_mm[k,0]*Wr[0][0] + X_test_mm[k,1]*Wr[0][1] + X_test_mm[k,2]*Wr[0][2] + X_test_mm[k,3]*Wr[0][3]+Ir  # predicted response for each day of test data using the polynomial
##        yhat_poly = np.array(X_test_mm[5,0])*Wr[0][0]# + X_test_mm.to_numpy()[k][1]*Wr[1] + X_test_mm.to_numpy()[k][2]*Wr[2] + X_test_mm.to_numpy()[k][3]*Wr[3]+Ir[0]  # predicted response for each day of test data using the polynomial
            Yhat_polynomial.append(np.asarray(yhat_poly)) #  Array of responses for one replication
##        y_test_loo_all.append(y_test_loo)
            
    #print('Yhat_polynomial',Yhat_polynomial)

##        j+=1      
##        Wr_all.append(Wr)
##        Ir_all.append(Ir)
        # print("Wr",Wr)
        # print("Ir",Ir)
    # print(len(y_test_mm))
##        #k=1
##        # yhat_regression = np.sum(X_test_mm[2,0]*Wr[0,0] + X_test_mm[2,1]*Wr[0,1] + X_test_mm[2,2]*Wr[0,2] + X_test_mm[2,3]*Wr[0,3]+Ir)
##        for k in range(len(y_test_mm)):  # For every iteration, use the regression equation to predict the response.  , 475
##            # yhat_reg=np.sum(X_test_mm[k,0]*Wr[0] + X_test_mm[k,1]*Wr[1] + X_test_mm[k,2]*Wr[2] + X_test_mm[k,3]*Wr[3]+Ir) #X_test_mm[k,0]*Wr[0,0] + X_test_mm[k,1]*Wr[0,1] + X_test_mm[k,2]*Wr[0,2] + X_test_mm[k,3]*Wr[0,3]+Ir[0]
##            # yhat_regression = np.sum(X_test_mm[k,0]*Wr[0] + X_test_mm[k,1]*Wr[1] + X_test_mm[k,2]*Wr[2] + X_test_mm[k,3]*Wr[3]+Ir)
##            yhat_poly = X_test_mm[k,0]*Wr[0][0] + X_test_mm[k,1]*Wr[0][1] + X_test_mm[k,2]*Wr[0][2] + X_test_mm[k,3]*Wr[0][3]+Ir  # predicted response for each day of test data using the polynomial
##            Yhat_polynomial.append(np.asarray(yhat_poly)) #  Array of responses for one replication
##            # yhat_poly = [].reshape(len(y_test_mm), -1)
    
    # Yhat_polynomial_all.append(np.asarray(Yhat_polynomial).reshape(len(y_test_mm), -1))
    
    
    # Wr_all_mean = np.mean(Wr_all, axis=0)       
    # Ir_all_mean = np.mean(Ir_all)
##
##    print('X_train_mm shape',X_train_mm.shape)
##    print('y_train_mm shape',y_train_mm.shape)
##    print('X_test_mm shape',X_test_mm.shape)
##    print('y_test_mm shape',y_test_mm.shape)
##    
##    print('Yhat_polynomial',Yhat_polynomial)
##    print('Yhat_polynomial type',type(np.asarray(Yhat_polynomial)))
##    print('Yhat_polynomial shape',np.asarray(Yhat_polynomial).shape)
##    print('Yhat_polynomial[0][:10]',Yhat_polynomial[0][:10])
##    print('Yhat_polynomial[0][:] length',len(Yhat_polynomial[0][:])) 
    Yhat_v_poly = np.reshape(Yhat_polynomial,(-1,100))  #np.reshape(Yhat_polynomial, (-1,nrep), order = 'F')  # F => Fortran order with index 1 (rows) changing fastest.
##    print('Yhat_v_poly head',Yhat_v_poly[:3])
##    print('Yhat_v_poly type',type(Yhat_v_poly))
##    print('Yhat_v_poly shape',Yhat_v_poly.shape)
##    print('Yhat_v_poly[3]',Yhat_v_poly[3])
##    print('Yhat_v_poly[3] length',len(Yhat_v_poly[3][0]))

    print('y_test_mm type',type(y_test_mm))
    print('y_test_mm shape',y_test_mm.shape)

    Yhat_poly_all = Yhat_v_poly
    #print(stats.describe(Yhat_poly_all))
    
    Poly_nobs, Poly_minmax, Poly_mean, Poly_variance, Poly_skewness, Poly_kurtosis = scipy.stats.describe(Yhat_poly_all, axis=0, ddof=1, bias=True, nan_policy='propagate')
    
    for n in range(nrep):
    # result.append(scipy.stats.linregress(y_test_mm, yhat_poly))  #y_test_mm[:], Yhat_polynomial[:,j]
        # res.append(scipy.stats.linregress(y_test_mm, Yhat_poly_all.T[n]))
        # inter.append(scipy.stats.linregress(y_test_mm[:,0], Yhat_poly_all[:,n]).intercept) # Y intercept of the polynomial
        # Rsq.append(scipy.stats.linregress(y_test_mm[:,0], Yhat_poly_all[:,n]).rvalue) #.rvalue
        Poly_sl.append(scipy.stats.linregress(y_test_mm[:,0], Yhat_poly_all[:,n]).slope) # slope of the regression line.reshape(-1,1).reshape(-1,1)
        Poly_pval.append(scipy.stats.linregress(y_test_mm[:,0], Yhat_poly_all[:,n]).pvalue)#
        # Poly_reg.append(linear_model.LinearRegression().fit(y_test_mm[:,0], Yhat_poly_all[:,n]))#
        Poly_rmse.append(sqrt(mean_squared_error(y_test_mm[:,0], Yhat_poly_all[:,n])))#
        Poly_r2_score.append(sklearn.metrics.r2_score(y_test_mm[:,0], Yhat_poly_all[:,n]))#.reshape(-1,1),Yhat_poly_all.T[n].reshape(-1,1))) #, fit_intercept=True

    print("Poly_rmse", Poly_rmse)
    print("Poly_r2", Poly_r2_score)
    
    columns = ['Min', 'Max', 'Mean', 'Variance', 'Slope', 'p-val', 'RMSE']
    len(columns)
    Poly_sum = np.concatenate((Poly_minmax[0], Poly_minmax[1], Poly_mean, Poly_variance, Poly_sl, Poly_pval, Poly_rmse), axis=0).reshape(-1,len(columns), order='F')   
    Poly_sum_df = pd.DataFrame(data = Poly_sum)
    Poly_sum_df.columns = columns
    #np.savetxt("C:/Users/mark_/mark_data/Output/Poly_sum_df.csv",  Poly_sum_df, delimiter=',')
    #np.savetxt("C:/Users/mark_/userdata/Output/yhat_regression_loo_100.csv", np.transpose(yhat_regression), delimiter=',')
    # plt.scatter(Poly_sum_df.index, Poly_sum_df.loc[:,'RMSE'])
    
    ## Chi-square test of the null hypothesis that there is no difference between the p-value or the slope of the linregress results.
    
    chisq, p = scipy.stats.chisquare(Poly_sum_df.loc[:,'p-val'])  #, Poly_sum_df.loc[:,'Slope']
    
    ## t-test of the null hypothesis that the difference in the means of 2 group means is zero.
    tstat, pval_t = scipy.stats.ttest_rel(Poly_sum_df.loc[:,'p-val'], Poly_sum_df.loc[:,'Slope'], axis=0, nan_policy='omit', alternative='two-sided')
    
    
    
    ## Multiple pairwise comparison of p-values. Null hypothesis is no difference in p-values.
    #  Returns "True" when the Null hypothesis of No difference (they are the same) can be rejected for the given alpha. The null hypothesis is that the observed difference is due to chance alone. 
    
    #print('Bonferroni - polynomial',statsmodels.stats.multitest.multipletests(Poly_sum_df.loc[:,'p-val'], alpha=0.05, method='bonferroni', is_sorted=False, returnsorted=True))
    # returns 1) reject null hypothesis?, 2) pvals_corrected, 3) alphacBonf (corrected alpha for Bonferroni method)
    

# print("MLP model validation test R2 is:" + str(reg_mlp.score(y_test_mm.reshape(-1,1),Yhat_v_m.reshape(-1,1))))   
    # Yhat_polynomial =[]    
     # print(Rsq[2].rvalue)
        # yhat_regression.append(yhat_poly)
            #yhat_poly = np.sum(X_test_mm[k,0]*Wr[0] + X_test_mm[k,1]*Wr[0] + X_test_mm[k,2]*Wr[0] + X_test_mm[k,3]*Wr[0]+Ir)
            # yhat_regression.append(np.asarray(yhat_reg))
            # Wr = 0.0   #  reset the model coefficients to zero before refitting to a new loo dataset.
            # Ir = 0.0    #  reset the model intercept to zero before refitting to a new loo dataset.
   
    # inter = np.array(Yhat_polynomial_all)# Array of values predicted (rows) X replicated loo runs (columns).  np.concatenate(yhat_reg[:], axis=1)#[0:-1]
    # # inter= np.asarray(np.concatenate(yhat_reg, axis=1))
    # Yhat_polynomial_2 = np.reshape(inter, (-1,len(y_test_mm))).T
    #     # yhat_reg = []
        
    
    # np.savetxt("C:/Users/mark_/userdata/Output/yhat_regression_loo_100.csv", np.transpose(yhat_regression), delimiter=',')
    # plt.scatter(yhat_r, y_test_mm, linestyle='-', c="k", s=1)
    # print(X_test_mm[k,1])
    # plt.figure(figsize=(16,6)) 
    # labels=["True values", "Predictions"]
    # plt.ylim(-0.05, 1)
    # plt.plot(y_test_mm)
    # # plt.plot(X_test_mm[:,3])
    # # n=0
    # for j in range(nrep):
    #     # n+=len(y_train_mm)
    #     # while n < len(y_train_mm): plt.plot(yhat_r[n+len(y_train_mm)])
    #     plt.plot(Yhat_polynomial[j])
    #     # plt.plot(yhat_regression[j])
    
    # plt.title('Polynomial model - Predicted and True values, testing ', fontsize='large') #MLP predicted and observed flow rate during testing
    # plt.legend(labels)
    # plt.show(); 
    # j=0
    # rms_poly = []
    # R2_poly = []
    # reg_poly = []
    # R2b_poly = []
    # Rsq = []
    # result= []#np.array(0)
    # plt.figure(figsize=(16,6)) 
    # labels=["True values", "Predictions"]
    # plt.ylim(-0.05, 1)
    # plt.plot(y_test_mm)
    # # plt.plot(X_test_mm[:,3])
    # # n=0
    # for j in range(nrep):
    #     # n+=len(y_train_mm)
    #     # while n < len(y_train_mm): plt.plot(yhat_r[n+len(y_train_mm)])
    #     # plt.plot(Yhat_polynomial[j])
    #     rms_poly.append(sqrt(mean_squared_error(y_test_mm, Yhat_polynomial[:,j])))
    #     result.append(scipy.stats.linregress(y_test_mm, Yhat_polynomial[:,j]))  #y_test_mm[:], Yhat_polynomial[:,j]
    #     Rsq.append(result[j].rvalue)
        # reg_poly = LinearRegression(fit_intercept=False).fit(y_test_mm, Yhat_polynomial[:,j])#, fit_inte
        # R2b_poly.append(reg_poly.score(y_test_mm, Yhat_polynomial[:,j]))#, fit_inte
        # plt.plot(Yhat_polynomial[:,j])
##    Yhat_v_poly = np.ravel(Yhat_polynomial, order='C')
##    Yhat_v_poly = Yhat_v_poly.T
####    numpy.rot90(m, k=1, axes=(0, 1))
####    Yhat_v_poly = np.asarray(Yhat_polynomial).flatten()
##    print('Yhat_v_poly shape', Yhat_v_poly.shape)
##    print('Yhat_v_poly[4][:]', Yhat_v_poly[4][:])
##    print('Yhat_v_poly type',type(Yhat_v_poly))
##    print('Yhat_v_poly shape',Yhat_v_poly.shape)
##
##    print('y_test_mm type',type(y_test_mm))
##    print('y_test_mm shape',y_test_mm.shape)
##    print('y_test_mm',y_test_mm)
##    y_test_mm = y_test_mm.astype(np.float)
##    print('y_test_mm',y_test_mm)
####    print('Yhat_polynomial shape',np.asarray(Yhat_polynomial).shape)
####    print('Yhat_v_poly', np.reshape(Yhat_v_poly,(-1,100)))# = pd.DataFrame(Yhat_polynomial).dropna() #np.asarray(Yhat_polynomial).astype('float')
####    print('Yhat_v_poly shape', np.reshape(Yhat_v_poly,(-1,100).shape))
##    
##    temp=[]
##    for m in range(nrep):
##        
##        temp.append(Yhat_v_poly[j][:])
##    temp = np.asarray(temp)
##    print('temp shape', temp.shape)
##    print('temp[1]', temp[1])
##    
##
##
##    R2_poly=[]
##    for j in range(nrep):
####        temp = 0
####        temp = Yhat_v_poly[j][:]
##        print('temp',temp)
##        slope, intercept, rvalue, pvalue, std_err = scipy.stats.linregress(y_test_mm[:,0],temp[:,0])# )  #R2_poly_rep
##        R2_poly.append(rvalue)
####        R2_poly.append(R2_poly_rep.rvalue)
##    print('R2_poly_rep',R2_poly)
    # result = np.array(result) 
    # Rsq = result[1].rvalue
    # # np.savetxt("C:/Users/mark_/Documents/02_Professional/03_Unil/SATW/MDPI_paper/MDPI_article/Data/yhat_regression_all_27", yhat_regression, delimiter=',')
    # np.savetxt("C:/Users/mark_/userdata/Output/y_test_mm_ns.csv", y_test_mm, delimiter=",", fmt="%10.4f") 
    # Sum = pd.DataFrame(data = yhat_regression)
    # Sum.T.describe() 
    
    #  Mean RMS
##    rms_poly_mean = np.mean(rms_poly) #, axis=0
##    print("Polynomial model mean rms error is: " + str(rms_poly_mean))
##    
##    #  Mean R2 using scipy stats linregress
##    Rsq_mean = np.mean(Rsq)
##    print("Polynomial model mean R-squared is: " + str(Rsq_mean))
##    
##    res = np.asarray(res)
    ##  Mean
    # R2_poly_mean = np.mean(R2_poly) #, axis=0
    # reg_poly.score(y_test_mm, Yhat_polynomial[:,2])
    #     # reg = LinearRegression(fit_intercept=False).fit(y_test_mm_t, np.asarray(Yhat_v)) 
    # reg_poly = LinearRegression(fit_intercept=False).fit(y_test_mm, Yhat_polynomial)#, fit_intercept=False forces intercept to zero, Yhat_polynomial
    # print("log trans, Validation test R2: %.4f" % reg_poly.score(y_test_mm, Yhat_polynomial))   #score is R2, the coefficient of determination
    
    # P =yhat_regression[1]# List of predictions
    # Yhat_polynomial_m_ns = np.mean(Yhat_polynomial, axis=1) # NON SHUFFLED data, mean predictions, 100 loo predictions
##    Yhat_polynomial_m = np.mean(Yhat_v_poly, axis=1) # SHUFFLED data, mean predictions, 100 loo predictions
    
    # plt.figure(figsize=(12,12)) 
    # labels=['fitted polynomial']
    # plt.ylim(0, 1.1)
    # # plt.plot(y_test_mm)
    # for i in range(100):  #len(X_test_mm)
    #     plt.plot(X_test_mm[i], inter[i] + Poly_sl[i]*X_test_mm[i], 'r')  ## Length X_test_mm = 459. Length inter and slope = 100.  No problem?
    # plt.scatter(y_test_mm, Yhat_polynomial_m)  #[:,0]
    # plt.title('Polynomial model - Predicted and True values, testing ', fontsize='large') #MLP predicted and observed flow rate during testing
    # plt.xlabel('True values')
    # plt.ylabel('Predictions') #Mean squared 
    # plt.legend(labels)
    # plt.show()

    ##    yl = np.array([0,1]) 
    ##    xl = np.array([0,1]) 
    ##
    ##    plt.figure(figsize=(16.2,10)) 
    ##    #labels=["True values", "Predictions"]
    ##    plt.ylim(0, 1)
    ##    plt.xlim(0, 1)
    ##    plt.scatter(y_test_mm, Yhat_polynomial_m)
    ##    plt.title('Polynomial model - Mean predicted versus True values, testing ', fontsize='large') #MLP predicted and observed flow rate during testing
    ##    #plt.legend(labels)
    ##    plt.plot(xl,yl,  c="k")
    ##    plt.savefig('C:/Users/mark_/Documents/02_Professional/03_Unil/SATW/MDPI_paper/complementary_information/figures/Fig_polynomial_cross_sh.svg', format="svg")
    ##    plt.show() 
    ##
    ##    plt.figure(figsize=(16,6)) 
    ##    labels=["True values", "Predictions"]
    ##    plt.plot(y_test_mm)
    ##    plt.plot(Yhat_polynomial_m)
    ##    plt.title('Polynomial model - Mean predicted and True values, testing ', fontsize='large') #MLP predicted and observed flow rate during testing
    ##    plt.legend(labels)
    ##    plt.savefig('C:/Users/mark_/Documents/02_Professional/03_Unil/SATW/MDPI_paper/complementary_information/figures/Fig_polynomial_time_sh.svg', format="svg")
    ##    plt.show() 
    # # T = y_test_mm[:,0].tolist()

    #print(Yhat_polynomial_m)
    
##    rms_poly = sqrt(mean_squared_error(y_test_mm, Yhat_polynomial_m))
##    print("Polynomial model validation test rms error is: " + str(rms))
####    
##    R2_poly_mod = scipy.stats.linregress(y_test_mm[:,0], Yhat_polynomial_m)
##    R2_poly = R2_poly_mod.rvalue
##    R2_poly = np.asarray(R2_poly).mean()
##    print("polynomial model validation test R2: %.2f" % R2.rvalue)
    
    # Yhat_polynomial_m_ns = Yhat_polynomial_m
    # y_test_mm_ns = y_test_mm

    Yhat_v_poly = Yhat_polynomial

    return  Yhat_v_poly#, y_test_loo_all#y_test_mm, , Yhat_polynomial_m,  Poly_sum_df, rms_poly, R2_poly#, rms_poly_mean, Rsq_mean,#Wr_all_mean, Ir_all_mean #R2_poly_mean  Yhat_polynomial_m_ns, 
    
####### USE WHEN RUNNING A SINGLE MODEL    ###########################3

def AFBR_MLP_model(X_train_mm, y_train_mm, X_test_mm, y_test_mm):  # , bch , do inputs, outputs, inputs_test, outputs_test    X_train_mm, y_train_mm, X_test_mm, y_test_mm
    # X_train_mm = tensorflow.data.Dataset.from_tensor_slices(X_train_mm) 
    # X_train_mm = tuple(map(tuple, inputs)) #np.stack(inputs).astype(None)  #.reshape(1071,4) 
    X_train_mm = X_train_mm.astype(np.float32)  # This works
    # # # testdf = pd.DataFrame(X_train_mm)
    # # X_train_mm = tf.convert_to_tensor(inputs)
    # X_train_mm = np.stack(X_train_mm.astype(float),0)  
    # X_train_mm = tuple(X_train_mm)res = ini_array.astype(np.float)
    # X_train_mm = X_train_mm.to_numpy(dtype=float)#  This creates an array of float64 that is accepted by the mlp
    # # # X_train_mm = tuple([tuple(x) for x in inputs])#np.stack(inputs).astype(None)  #.reshape(1071,4) 
    # # # y_train_mm = [tuple(x) for x in outputs]#tuple([tuple(x) for x in outputs])#.tolist()]
    # # y_train_mm = np.stack(outputs)
    y_train_mm = y_train_mm.astype(np.float32)# This works
    # y_train_mm = tuple(map(tuple, outputs)) #[tuple(x) for x in outputs.tolist()]
    # X_test_mm = tuple(map(tuple, inputs_test)) #[tuple(x) for x in inputs_test.tolist()]
    # y_test_mm = tuple(map(tuple, outputs_test)) #[tuple(x) for x in outputs_test.tolist()]
    # X_test_mm = tuple([tuple(x) for x in X_test_mm]) #[tuple(x) for x in inputs_test.tolist()]
    # X_test_mm = np.stack(X_test_mm,0) 
    # X_test_mm = X_test_mm.to_numpy(dtype=float)#This creates an array of float64 that is accepted by the mlp
    X_test_mm = X_test_mm.astype(np.float32)# This works
    # y_test_mm = tuple([tuple(x) for x in outputs_test]) #[tuple(x) for x in outputs_test.tolist()]
    y_test_mm = y_test_mm.astype(np.float32)# This works
    # # print(X_train_mm.shape)
    # # input_tensor = X_train_mm
    # np.savetxt('/scratch/mmccormi1/y_test_mm.csv', y_test_mm, delimiter=',')
    # inputs = np.arange(100).reshape(-1,4)
    # [tuple(x) for x in inputs.tolist()]
    # print(X_train_mm[:])
      
    
# Call from the satw_all file
# Use a single hyperparameter variable or pass list of hyperparameter variables
# Reload a saved NN model(satw_all) or save the newly built model in h5 format (see line 270)   
    b =4  # bch #bch# batch_size = 4 works best
    node=256#256#32 # 512 V1 = 16 nodes 32*4 = 128 2048
    lr = 0.0001  # 0.00001 works best with RMSProp. 0.000001 works best with Adam
    ep = 100#400  # V1 best = 250
    #b1 = 0.9 # Used in Adam optimizer. Exponential decay rate for the first moment estimates. Default = 0.9
    do = 0.05#do#0.01 #0.05 # drop out rate
    #elu = keras.layers.ELU(alpha=0.9)
    rho = 0.99 #Discounting factor for the history/coming gradient. Defaults to 0.9.
    mom = 0.05#0.0 #Defaults to 0.0.
    
    initializer = tensorflow.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1, seed = None) #, seed=42, glorot_uniform(seed=42)  RandomNormal(mean=0.0, stddev=sdv, seed=42)
    # tf.keras.activations.relu(x, alpha=0.0, max_value=None, threshold=0)
   
    # sess.graph contains the graph definition; that enables the Graph Visualizer.

    # file_writer = tf.summary.FileWriter('\\Anaconda3\\envs\\Tensorflow_v2\\Mark\\', sess.graph)
    
    # with tf.Session() as sess:
    #     writer = tf.summary.FileWriter("\\Anaconda3\\envs\\Tensorflow_v2\\Mark\\", sess.graph)
    ######## SEQUENTIAL MODEL ###############
    
    ####Multilayer model
    AFBR = tensorflow.keras.Sequential()  # keras.
    AFBR.add(tensorflow.keras.layers.InputLayer(input_shape=(4,)))  #   keras.layers.Dense(4, input_shape=(4,)) keras.layers.InputLayer.  This works: AFBR.add(tensorflow.keras.Input(shape=(4,)))
    AFBR.add(tensorflow.keras.layers.Dense(units=node, activation='relu', use_bias=True, kernel_initializer=initializer, bias_initializer = 'zeros'))#,

    
    ###kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4)
    # AFBR.add(layers.Dense(units=6*node, activation= 'relu', use_bias=True, kernel_initializer=initializer,
    # bias_initializer = 'zeros'))#,  activity_regularizer=l1(0.1))) # V1 use_bias = True, , kernel_regularizer='l1'name ="Dense layer 1"'zeros'"glorot_uniform"(seed=None) relu,, input_dim=(1),input_shape=(step,)  , input_shape=(24,8)units=2, , AFBR.add(Dropout(0.1, noise_shape=None, seed=None)),  "glorot_uniform"
    # AFBR.add(layers.Dense(units=12*node, activation= 'relu', use_bias=True, bias_initializer = 'zeros'))# , kernel_regularizer=l1(0.01), activity_regularizer=l2(0.01), kernel_regularizer='l1', activity_regularizer="l1")) #,, input_dim=(1),input_shape=(step,)  , input_shape=(24,8)units=2, , 
    # AFBR.add(Dropout(do))
    # AFBR.add(layers.Dense(units=24*node, activation= 'relu', use_bias=True, kernel_initializer=initializer, bias_initializer = 'zeros'))#, kernel_regularizer=l1(0.01) , activity_regularizer=l2(0.01)))#, kernel_regularizer='l1', activity_regularizer="l1")) #
    # AFBR.add(Dropout(do))
    # AFBR.add(layers.Dense(units=48*node, activation= 'relu', use_bias=True, kernel_initializer=initializer, bias_initializer = 'zeros'))#, kernel_regularizer=l1(0.01), kernel_regularizer=l1(0.01) , activity_regularizer=l2(0.01)))#, kernel_regularizer='l1', activity_regularizer="l1"))
    # AFBR.add(Dropout(do))
    # AFBR.add(layers.Dense(units=96*node, activation='relu', use_bias=True, kernel_initializer=initializer, bias_initializer = 'zeros'))#, kernel_regularizer=l1(0.01) , activity_regularizer=l2(0.01)))#, kernel_regularizer='l1', activity_regularizer="l1"))
    # AFBR.add(Dropout(do))
    # AFBR.add(layers.Dense(units=192*node, activation='relu', use_bias=True, kernel_initializer = initializer, bias_initializer = 'zeros'))#, kernel_regularizer='l1', activity_regularizer="l1"))#, bias_initializer = 'zeros'))
    # Use the following layers to build the 9 layer MLP
    # AFBR.add(Dropout(do))
    # AFBR.add(tensorflow.keras.layers.Dense(units=4*node, activation= 'relu', use_bias=True, kernel_initializer=initializer, bias_initializer = 'zeros'))#, kernel_regularizer=l1(0.01), kernel_regularizer=l1(0.01) , activity_regularizer=l2(0.01)))#, kernel_regularizer='l1', activity_regularizer="l1"))
    # AFBR.add(Dropout(do))
    #### Final 4 hidden layer model  #####
    AFBR.add(tensorflow.keras.layers.Dense(units=node, activation='relu', use_bias=True, kernel_initializer=initializer, bias_initializer = 'zeros'))#, kernel_regularizer=l1(0.01) , activity_regularizer=l2(0.01)))#, kernel_regularizer='l1', activity_regularizer="l1"))
    AFBR.add(Dropout(do))
    AFBR.add(tensorflow.keras.layers.Dense(units=node, activation='relu', use_bias=True, kernel_initializer = initializer, bias_initializer = 'zeros'))#, kernel_regularizer='l1', activity_regularizer="l1"))#,
    AFBR.add(Dropout(do))
    AFBR.add(tensorflow.keras.layers.Dense(units=node, activation= 'relu', use_bias=True, kernel_initializer=initializer, bias_initializer = 'zeros'))#, kernel_regularizer=l1(0.01), kernel_regularizer=l1(0.01) , activity_regularizer=l2(0.01)))#, kernel_regularizer='l1', activity_regularizer="l1"))
    AFBR.add(Dropout(do))
    AFBR.add(tensorflow.keras.layers.Dense(units=node, activation='relu', use_bias=True, kernel_initializer=initializer, bias_initializer = 'zeros'))#, kernel_regularizer=l1(0.01) , activity_regularizer=l2(0.01)))#, kernel_regularizer='l1', activity_regularizer="l1"))
##    AFBR.add(Dropout(do))
##    AFBR.add(tensorflow.keras.layers.Dense(units=node, activation='relu', use_bias=True, kernel_initializer = initializer, bias_initializer = 'zeros'))#, kernel_regularizer='l1', activity_regularizer="l1"))#,
    # AFBR.add(Dropout(do))
    # AFBR.add(tensorflow.keras.layers.Dense(units=node, activation= 'relu', use_bias=True, kernel_initializer=initializer, bias_initializer = 'zeros'))#, kernel_regularizer=l1(0.01), kernel_regularizer=l1(0.01) , activity_regularizer=l2(0.01)))#, kernel_regularizer='l1', activity_regularizer="l1"))
    # AFBR.add(Dropout(do))
    # AFBR.add(tensorflow.keras.layers.Dense(units=node, activation='relu', use_bias=True, kernel_initializer=initializer, bias_initializer = 'zeros'))#, kernel_regularizer=l1(0.01) , activity_regularizer=l2(0.01)))#, kernel_regularizer='l1', activity_regularizer="l1"))
    # AFBR.add(Dropout(do))
    # AFBR.add(tensorflow.keras.layers.Dense(units=node, activation='relu', use_bias=True, kernel_initializer = initializer, bias_initializer = 'zeros'))#, kernel_regularizer='l1', activity_regularizer="l1"))#,
    # AFBR.add(Dropout(do))
    # AFBR.add(tensorflow.keras.layers.Dense(units=node, activation= 'relu', use_bias=True, kernel_initializer=initializer, bias_initializer = 'zeros'))#, kernel_regularizer=l1(0.01), kernel_regularizer=l1(0.01) , activity_regularizer=l2(0.01)))#, kernel_regularizer='l1', activity_regularizer="l1"))
    # AFBR.add(Dropout(do))
    # AFBR.add(tensorflow.keras.layers.Dense(units=node, activation='relu', use_bias=True, kernel_initializer=initializer, bias_initializer = 'zeros'))#, kernel_regularizer=l1(0.01) , activity_regularizer=l2(0.01)))#, kernel_regularizer='l1', activity_regularizer="l1"))
    # AFBR.add(Dropout(do))
    # AFBR.add(tensorflow.keras.layers.Dense(units=node, activation= 'relu', use_bias=True, kernel_initializer=initializer, bias_initializer = 'zeros'))#, kernel_regularizer=l1(0.01), kernel_regularizer=l1(0.01) , activity_regularizer=l2(0.01)))#, kernel_regularizer='l1', activity_regularizer="l1"))
    # AFBR.add(Dropout(do))
    # AFBR.add(tensorflow.keras.layers.Dense(units=node, activation='relu', use_bias=True, kernel_initializer=initializer, bias_initializer = 'zeros'))#, kernel_regularizer=l1(0.01) , activity_regularizer=l2(0.01)))#, kernel_regularizer='l1', activity_regularizer="l1"))
    
    # AFBR.add(Dropout(do))
   
    # Use the following layers to build the tapered MLP
    # AFBR.add(Dropout(do))
    # AFBR.add(layers.Dense(units=96*node, activation='relu', use_bias=True, kernel_initializer=initializer, bias_initializer = 'zeros'))
    # AFBR.add(Dropout(do))
    # end tapered
    AFBR.add(tensorflow.keras.layers.Dense(units=1)) #,, activation='linear',input_dim=(1),input_shape=(step,)  , input_shape=(24,8)units=2, , 
    
    print(AFBR.summary())
    
    
   
     ######## FUNCTIONAL API ############### 
    # input_layer = tensorflow.keras.Input(shape=(4,))
    # # Dense = layers.Dense(units=6*node, activation= 'relu', use_bias=True, kernel_initializer=initializer, bias_initializer = 'zeros' )
   
    # x = layers.Dense(units=6*node, activation = activations.relu, use_bias=True, kernel_initializer=initializer, bias_initializer = 'zeros' )(input_layer)  #activation = 'relu'
    # # x = layers.Dense(units=12*node, activation = activations.relu, use_bias=True, kernel_initializer=initializer, bias_initializer = 'zeros' )(x)
    # # x = layers.Dropout(do)(x)
    # # x = layers.Dense(units=24*node, activation = activations.relu, use_bias=True, kernel_initializer=initializer, bias_initializer = 'zeros' )(x)
    # # x = layers.Dropout(do)(x)
    # # x = layers.Dense(units=48*node, activation = activations.relu, use_bias=True, kernel_initializer=initializer, bias_initializer = 'zeros' )(x)
    # # x = layers.Dropout(do)(x)
    # # x = layers.Dense(units=96*node, activation = activations.relu, use_bias=True, kernel_initializer=initializer, bias_initializer = 'zeros' )(x)
    # # x = layers.Dropout(do)(x)
    # # x = layers.Dense(units=192*node, activation = activations.relu, use_bias=True, kernel_initializer=initializer, bias_initializer = 'zeros' )(x)
    # output_layer = layers.Dense(1)(x)
    
    # AFBR = tensorflow.keras.Model(inputs=input_layer, outputs=output_layer, name="AFBR_functional_API")

    # print(AFBR.summary())
    
    # print(inputs.shape)
    # print(outputs.shape)
    # print(inputs.dtype)
    # print(outputs.dtype)
    # prediction from training data
   
    
    metric= tensorflow.keras.metrics.MeanAbsoluteError(name='mean_absolute_error')
   
    
   # AFBR.compile(optimizer=keras.optimizers.Adam(learning_rate=lr, beta_1=b1, beta_2=0.999, epsilon=0.00000001), loss="mse",  metrics = [metric]) #default:Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=0.0000001)keras.metrics.mse()loss = 'mean_squared_logarithmic_error', "RMSprop", metrics=['accuracy'], optimizer= sgd, optimizer,'adam'  metrics = mean_squared_error
    #Centered = True -> gradiants are normalized about the variance of the gradianet. If False, by uncentered variance. The first moment is the mean and the second moment about the mean is the sample variance
    # AFBR.compile(optimizer=keras.optimizers.SGD(learning_rate=0.01, momentum=mom, nesterov= False), loss="mse",  metrics = [metric])
   
    # Tensorboard
    # tf.reset_default_graph()   Paste in the Spyder console
    # g = tf.graph()
    # Clear any logs from previous runs
    # rm -rf \\logs\\
 ######### Save the MLP model  #####
 ## V2 = All 9 L9 experiments.
 ## V3 = L9 with Experiment 2 removed (=> 8 experiments)
    
    # AFBR.save("/scratch/mmccormi1/AFBR_mlp_v2.h5")   
    # print("Saved model to HPC")
    
    # AFBR.save('C:/Users/mark_/Anaconda3/envs/Tensorflow_v2/Mark/AFBR_mlp_v3.h5')   
    # print("Saved model to disk")
    
    # AFBR.save("/scratch/mmccormi1/AFBR_mlp_v3", save_format='tf')#   Use in tf2
    # AFBR_t = tf.keras.models.load_model('C:/Users/mark_/Anaconda3/envs/Tensorflow_v1/Mark/AFBR_mlp')
     
        
    #"C:\Users\mark_\Anaconda3\envs\Tensorflow_v2\Lib\site-packages\logs\scalars\20200805-144545"
    # file_writer = tf.summary.FileWriter('tflearn_logs\\') #Anaconda3\\envs\\Tensorflow_v1\\Mark\\tflearn_logs\\
    # file_writer.add_graph(sess.graph)
    # log_dir = 'C:\\Users\mark_\Anaconda3\envs\Tensorflow_v2\Lib\site-packages\logs\scalars' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  ##'logs\\scalars\\' \\scalars\\  'C:\\Anaconda3\\envs\\Tensorflow_v1\\Mark\\tflearn_logs\\' Must use 2 back slashes
    # log_dir = os.path.join('logs','fit', datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    # print(log_dir)
    # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=False, write_images=True)  #, histogram_freq=1 update_freq='epoch',
    # tensorboard --logdir '\\Anaconda3\\envs\\Tensorflow_v2\\Mark\\tflearn_logs'    paste to the Anaconda command line
    # callbacks = [
    #     keras.callbacks.ModelCheckpoint(
    #         filepath='C:/Users/mark_/Anaconda3/envs/Tensorflow_v2/Mark{epoch}',
    #         save_freq='epoch')
    #     ]
    
    
    ###### Get MLP training and test loss histories 
    AFBR.compile(optimizer=tensorflow.keras.optimizers.RMSprop(learning_rate=lr, rho=rho, momentum = mom, epsilon=0.0000001, centered= True), loss=tensorflow.keras.losses.MeanSquaredError(),  metrics = [metric]) #,"mse" clipnorm=1.0 default: RMSprop(lr=0.001, rho=0.9, momentum=0.0, epsilon = 0.0000001, centered = False gives slightly better rmse) 
    history = AFBR.fit(X_train_mm, y_train_mm, validation_data = (X_test_mm, y_test_mm), batch_size= int(b), epochs=ep,  shuffle = False, verbose=0)  # , inputs_test, outputs_test,  shuffle = True to shuffle before splitting., verbose=3, callbacks=[tensorboard_callback], tensorboard_callback, ICANN paper: 250 epochs    
    
    # history_t = AFBR_t.fit(X_train_mm, y_train_mm, validation_data = (X_test_mm, y_test_mm), epochs=100, batch_size= int(b), shuffle = True, verbose=1) 
    # history = AFBR.fit(X_train, y_train, epochs=100, batch_size=int(step), verbose=1)
     
    test_scores = AFBR.evaluate(X_test_mm, y_test_mm , verbose=0) #X_test_mm, y_test_mm
    print("Training loss (MSE):", history.history['loss'][0])
    print("Test loss (MSE):", history.history['val_loss'][1])
    print("Mean absolute error:", test_scores[0])
    
    
    #####  Get AutiInit training and test loss histories
    # from autoinit import AutoInit
    # training_model = AutoInit().initialize_model(training_model)
    
    # training_model = AutoInit().initialize_model(AFBR)
    
    # training_model.compile(optimizer=tensorflow.keras.optimizers.RMSprop(learning_rate=lr, rho=rho, momentum = mom, epsilon=0.0000001, centered= True), loss=tensorflow.keras.losses.MeanSquaredError(),  metrics = [metric]) #,"mse" clipnorm=1.0 default: RMSprop(lr=0.001, rho=0.9, momentum=0.0, epsilon = 0.0000001, centered = False gives slightly better rmse) 
    # history = training_model.fit(X_train_mm, y_train_mm, validation_data = (X_test_mm, y_test_mm), batch_size= int(b), epochs=ep,  shuffle = False, verbose=0)  # , inputs_test, outputs_test,  shuffle = True to shuffle before splitting., verbose=3, callbacks=[tensorboard_callback], tensorboard_callback, ICANN paper: 250 epochs    
   
    # test_scores = training_model.evaluate(X_test_mm, y_test_mm , verbose=0) #X_test_mm, y_test_mm
    # print("Training loss (MSE):", history.history['loss'][0])
    # print("Test loss (MSE):", history.history['val_loss'][1])
    # print("Mean absolute error:", test_scores[0])
    
    # from autoinit import AutoInitVisualizer
    # AutoInitVisualizer().visualize(training_model)
    
   
   
    
    # test_mse = test_scores[0]
    # print('input shape:%.1f%%', inputs)
    # print('input test shape:%.1f%%', inputs_test)
    
    ########    MDPI figure 4    ###########
    # MLP loss
    #plt.figure(figsize=(16.2,10))
    fig, axs = plt.subplots(figsize=(16.2,10))
##    plt.plot(history.history['loss'], c="g", linestyle="-", linewidth = 4, label='training_loss') #, 'r'train_history
##    plt.plot(history.history['val_loss'], c="k", linestyle="-", linewidth = 2, label='test_loss') #, 'b'val_history, val_loss
    axs.plot(history.history['loss'], c="g", linestyle="-", linewidth = 4, label='training_loss') #, 'r'train_history
    axs.plot(history.history['val_loss'], c="k", linestyle="-", linewidth = 2, label='test_loss') #, 'b'val_history, val_loss
    axs.tick_params(axis="y", labelsize=16, direction="in")
    axs.set_ylabel("Mean Squared Error", fontsize=24)
    axs.tick_params(axis="x", labelsize=16, direction="in")
    axs.set_xlabel("Epoch", fontsize=24)#
    plt.legend(loc="upper right")
    plt.ylim(0.00, 0.025)
##    plt.xlabel('Epoch')
##    plt.ylabel('Mean Squared Error') #Mean squared 
    #plt.title('MLP model- Loss during training and testing', fontsize='large') #and testing
##    plt.savefig('C:/Users/mark_/Documents/02_Professional/03_Unil/SATW/MDPI_paper/Journal communication/Sustainability/Round two special issue/Figures_originals/Fig_mlp_loss.svg', format="svg", bbox_inches="tight", dpi=300)
##    plt.savefig('C:/Users/mark_/Documents/02_Professional/03_Unil/SATW/MDPI_paper/Journal communication/Sustainability/Round two special issue/Figures_originals/Fig_mlp_loss.pdf', format="pdf", bbox_inches="tight", dpi=300)
    # plt.savefig('/scratch/mmccormi1/mlp_training.svg', format="svg")"C:\Users\mark_\Documents\02_Professional\03_Unil\SATW\MDPI_paper\Journal communication\Sustainability\Round two special issue\Figures_originals\Fig_mlp_loss.svg"
    plt.show()

##fig, axs = plt.subplots(figsize=(16.2,10))#16.2
##    
####    plt.figure(figsize=(12,10))
####    plt.plot(test['CVref'], linestyle=" ", marker='D', c="r")  # Reference experience
####    plt.plot(test['CV_n'], linestyle=" ", marker='.', c="b") # Derived values   
####    axs.plot(new[:(len(y)+1)], linestyle=" ", marker='D', c="r",  markersize=5, label=my_label["Reference experiment"])  # Reference experiencekind='scatter',ax=axs,
####    axs.plot(new[(len(y)+1):], linestyle=" ", marker='.', c="b", markersize=5, label=my_label["Derived values"])
##
##    axs.plot(y, linestyle=" ", marker='D', c="r",  markersize=5, label=my_label["Reference experiment"])  # Reference experiencekind='scatter',ax=axs,
##    axs.tick_params(axis="y", labelsize=16, direction="in")
##    axs.set_ylabel("Calorific value reduction [J.liter$^{-1}$]", fontsize=24)
##    
##    axs.plot(CV_red_n, linestyle=" ", marker='.', c="b", markersize=5, label=my_label["Derived values"])
##    axs.tick_params(axis="x", labelsize=16, labelrotation=-90, direction="in")
##    axs.set_xlabel("Date", fontsize=24)#
##    plt.subplots_adjust(bottom=0.175)
##    #axs.xaxis.set_minor_locator(AutoMinorLocator())
##    
##    #ax.plot(CV_red_n.iloc[:,0], linestyle=" ", marker='.', c="b") # Derived values
##    #plt.xlabel("Date", fontsize=20)
##    #plt.xticks( rotation='vertical')#x, labels[::2],
##    #plt.xticks(xticklabels[::10], rotation='vertical')
##    # ax.set_xticks(test[::10])
##    #axs.set_xticklabels(xticklabels[::10], rotation = 90)
##    #plt.ylabel("Calorific value reduction [J.liter$^{-1}$]", fontsize=20)
##    # plt.ylim([0.05,0.5])
##    plt.legend(loc="upper right")  #, numpoints = 1
##    #plt.title('Experimental results', fontsize='large')
##    #plt.savefig('C:/Users/mark_/Documents/02_Professional/03_Unil/SATW/MDPI_paper/Journal communication/Sustainability/Round two special issue/Figures_originals/F1_experiment.svg', format="svg")
##    plt.show()


    
    # # # plt.savefig('C:/Users/mark_/Anaconda3/envs/tensorflow_env/mark/satw/B_AFBR_loss.svg', dpi=None, facecolor='w', edgecolor='w',
    # # #        orientation='portrait', papertype=None, format=None,
    # # #        transparent=False, bbox_inches='tight', pad_inches=0.1,
    # # #        frameon=None, metadata=None)  
    # # mlp accuracy
    # # print(metric.result().numpy())
    # # plt.figure(figsize=(8,4))
    # # # plt.title('MLP training loss', fontsize='large') #and testing
    # # plt.plot(history.history['loss'], c="g", linestyle="-", label='training_loss') #, 'r'train_history
    # # plt.plot(history.history['val_loss'], c="m", linestyle="-", label='test_loss') #, 'b'val_history, val_loss
    # # # plt.plot(history_t.history['loss'], label='training_loss-2') #, 'r'train_history
    # # # plt.plot(history_t.history['val_loss'], label='test_loss-2') #, 'b'val_history
    # # plt.legend()
    # # plt.xlabel('Epoch')
    # # plt.ylabel('Mean Squared Error') #Mean squared 
    # # # plt.savefig('C:/Users/mark_/Documents/02_Professional/03_Unil/SATW/MDPI_paper/MDPI_article/MDPI_article_v03/figures/F4_mlp_loss.svg', format="svg")
    # # plt.show() 
    
### Testing during model building 
    Yhat = AFBR.predict([X_train_mm]).reshape(-1,1)
    # np.savetxt("C:/Users/mark_/Documents/02_Professional/03_Unil/SATW/MDPI_paper/MDPI_article/MDPI_article_v03/figures/Yhat_train_mlp_noshuffle", Yhat, delimiter=',')
    # np.savetxt('/scratch/mmccormi1/Yhat.csv', Yhat, delimiter=',')
    # print("Yhat shape is: " + str(Yhat.shape))
    # prediction from validation test data
    Yhat_v_mlp = AFBR.predict([X_test_mm]).reshape(-1,1)##Yhat_v = predictions during the validation run. steps=1, batch_size=4,, verbose=1

### AutoInit testing during model building 
    # Yhat = training_model.predict([X_train_mm]).reshape(-1,1)
    # np.savetxt("C:/Users/mark_/Documents/02_Professional/03_Unil/SATW/MDPI_paper/MDPI_article/MDPI_article_v03/figures/Yhat_train_mlp_noshuffle", Yhat, delimiter=',')
    # np.savetxt('/scratch/mmccormi1/Yhat.csv', Yhat, delimiter=',')
    # print("Yhat shape is: " + str(Yhat.shape))
    # prediction from validation test data
    # Yhat_v_m = training_model.predict([X_test_mm]).reshape(-1,1)#Yhat_v = predictions during the validation run. steps=1, batch_size=4,, verbose=1
       

    # Save predictions to the PC 
    # np.savetxt('C:/Users/mark_/userdata/Output/Yhat_v_mlp_4a.csv', Yhat_v, delimiter=',')  # 4a, shuffled data, 1-layer, SEE FUNCTION BELOW
    # np.savetxt("C:/Users/mark_/userdata/Output/Yhat_v_mlp_4b.csv", Yhat_v, delimiter=',')  # 4b, unshuffled data, 6-layer mlp
    # np.savetxt("C:/Users/mark_/userdata/Output/Yhat_v_mlp_4c.csv", Yhat_v, delimiter=',')  # 4c, shuffled data, 6-layer mlp
    # np.savetxt("C:/Users/mark_/userdata/Output/Yhat_v_1l.csv", Yhat_v, delimiter=',')  # 1 layer result to add to archiopti
    
    # Save predictions to the CLUSTERS 
    # np.savetxt('/scratch/mmccormi1/Yhat_v_mlp_b.csv', Yhat_v, delimiter=',')  # b, shuffled data, 1-layer, SEE FUNCTION BELOW
    # np.savetxt("/scratch/mmccormi1/Yhat_v_mlp_c.csv", Yhat_v, delimiter=',')  # c, unshuffled data, 6-layer mlp
    # np.savetxt("/scratch/mmccormi1/Yhat_v_mlp_d.csv", Yhat_v, delimiter=',')    # d, shuffled data, 6-layer mlp
    # np.savetxt("/scratch/mmccormi1/Yhat_v_mlp_4d.csv", Yhat_v, delimiter=',')  # e, shuffled data, 12-layer mlp, gradual reduction      
    
    # mse_mlp = mean_squared_error(y_test_mm, Yhat)  #outputs,predictions
    # print('MLP MSE:%.1f%%', mse_mlp)
    # mse_mlp_t = mean_squared_error(y_test_mm, Yhat_t)  #outputs,predictions
    # print('MLP_t MSE:%.1f%%', mse_mlp_t)
    # np.savetxt('C:/Users/mark_/Anaconda3/envs/Tensorflow_v1/Mark/mse_mlp.csv')
    # MSE.to_csv('C:/Users/mark_/Anaconda3/envs/Tensorflow_v1/Mark/mse_mlp.csv', sep=';', index=False, decimal=',')
    #Yhat_t = scaler.inverse_transform(X_test_mm) #Yhat.reshape (-1,1)

       
    
##### Figures      #######################
##    y3 = np.array([0,1])#len(y_test_mm)]) 
##    x3 = np.array([0,1])#len(y_test_mm)])
##    
##    plt.figure(figsize=(14.1,10)) 
##    #labels=["True values", "Predictions"]
##    plt.ylim(0, 1)
##    plt.xlim(0, 1)
##    plt.scatter(y_test_mm, Yhat_v_mlp)
##    plt.plot(x3,y3, c='k')
##    plt.title('MLP - Predicted versus True values, testing ', fontsize='large') #MLP predicted and observed flow rate during testing
##    #plt.legend(labels)
##    plt.xlabel('CV reduction (MinMax scaled)') #Measurement sequence number
##    plt.ylabel('CV reduction (MinMax scaled)')
##    #plt.savefig('C:/Users/mark_/Documents/02_Professional/03_Unil/SATW/MDPI_paper/complementary_information/figures/Fig_mlp_T_vs_P.svg', format="svg")
##    plt.show()

    

##    plt.figure(figsize=(15,10)) 
##    labels=["True values", "Predictions"]
##    plt.plot(y_test_mm)
##    plt.plot(Yhat_v_mlp)
##    plt.title('MLP model - Predicted and True values, testing ', fontsize='large') #MLP predicted and observed flow rate during testing
##    plt.legend(labels)
##    plt.xlabel('Sample [Day]') #Measurement sequence number
##    plt.ylabel('CV reduction (MinMax scaled)')
##    #plt.savefig('C:/Users/mark_/Documents/02_Professional/03_Unil/SATW/MDPI_paper/complementary_information/figures/Fig_mlp_T_and_P_time.svg', format="svg")
##    plt.show() 
    
    # # plt.figure(figsize=(16,6)) 
    # # plt.plot(y_test_mm-Yhat_v)
    # # plt.xlabel('Day') #Measurement sequence number
    # # plt.ylabel('Raw error (True-predicted)')
    # # plt.title('MLP model - Error during testing ', fontsize='large') #MLP predicted and observed flow rate during testing
    # # plt.savefig('/scratch/mmccormi1/mlp_train_acc.svg', format="svg")
    # # plt.show();    
    # # mse_mlp = mean_squared_error(y_test_mm, Yhat_v)  #outputs,predictions
    # # print('MLP MSE:%.1f%%', mse_mlp)
    # #Graph predicted and observed
    # plt.figure(figsize=(16,6)) 
    # #plt.plot(Yhat_t[:,0], 'r')
    # #plt.plot(Yhat_t[:,1], 'c')
    # #plt.plot(Yhat_t[:,2], 'g')
    # #plt.plot(Yhat_t[:,3])#, 'y'
    # plt.plot(y_test_mm, c="g", linestyle="-", label = 'True outputs (Test data)')#, 'y'
    # plt.plot(Yhat_v, c="k", linestyle="-", label ='Predictions (Test data)' )#, 'y'[:,0]
    # # plt.plot(Yhat_t, label= 'Predictions (Test-2 data)') #, 'y'[:,0]
    # #plt.plot(y_test)#, 'y'
    # plt.legend()#['True outputs (Test data)', 'Predictions (Test data)', 'Predictions (Test-2 data)'])
    # plt.xlabel('Day') #Measurement sequence number
    # plt.ylabel('Daily CV reduction (MinMax scaled)')
    # plt.title('MLP model - accuracy during testing ', fontsize='large') #MLP predicted and observed flow rate during testing
    # # plt.savefig('/scratch/mmccormi1/mlp_test_acc.svg', format="svg")
    # # # plt.savefig('C:/Users/mark_/Documents/02_Professional/03_Unil/SATW/MDPI_paper/MDPI_article/MDPI_article_v03/figures/F3_mlp_6-layer_noshuffle.svg', format="svg")#, dpi=None, facecolor='w', edgecolor='w',
    # # #     orientation='portrait', papertype=None, format=None,
    # # #     transparent=False, bbox_inches='tight', pad_inches=0.1,
    # # #     frameon=None, metadata=None) 
    # # plt.show();
    
    rmse_mlp = sqrt(mean_squared_error(y_test_mm, Yhat_v_mlp))
    #print("MLP model validation test rms error is: " + str(test_rmse))

##    print('y_test_mm',y_test_mm.reshape(-1,1))
##    print('Yhat_v_mlp',Yhat_v_mlp[:,0])
    #### Atention. Might need to comment when running NN_DC
    #slope, intercept, r_value, p_value = stats.linregress(y_test_mm.reshape(-1,1), Yhat_v_mlp)  #  result ( slope, intercept, rvalue, pvalue, stderr y_test_mm[:,0], Yhat_v_mlp[:,0]scipy.
    #slope, intercept, r_value, p_value, std_err = stats.linregress(X[:,0], Y[:,0])
    #print("MLP model validation test R2: %.2f" % R2_mlp.rvalue) 
##    R2_poly=[]
##    for j in range(nrep):
##        R2_poly_rep = scipy.stats.linregress(y_test_mm[:,0], Yhat_poly_all[:,j])
##        R2_poly.append(R2_poly_rep.rvalue)
    
    # corr, p_value = scipy.stats.pearsonr(y_test_mm, Yhat_v)
    # print('Pearsons correlation: %.3f' % corr)
    # print('Pearsons p-value: %.3f' % p_value)
    # print("R`2 is: " + str(Rsqr))
    # plt.figure(figsize=(16,6)) 
    # #plt.plot(Yhat_t[:,0], 'r')
    # #plt.plot(Yhat_t[:,1], 'c')
    # #plt.plot(Yhat_t[:,2], 'g')
    # #plt.plot(Yhat_t[:,3])#, 'y'
    # plt.plot(y_test_mm, label = 'True outputs (Test data)')#, 'y'
    # # plt.plot(Yhat, label ='Predictions (Test data)' )#, 'y'[:,0]
    # plt.plot(Yhat_t, label= 'Predictions (Test-2 data)') #, 'y'[:,0]
    # #plt.plot(y_test)#, 'y'
    # plt.legend()#['True outputs (Test data)', 'Predictions (Test data)', 'Predictions (Test-2 data)'])
    # plt.xlabel('Day') #Measurement sequence number
    # plt.ylabel('Daily CH4 production (MinMax scaled)')
    # plt.title('MLP model - accuracy during testing ', fontsize='large') #MLP predicted and observed flow rate during testing
    # # plt.savefig('Anaconda3/envs/tensorflow_env/mark/satw/C_AFBR_pred_obs.png', dpi=None, facecolor='w', edgecolor='w',
    # #     orientation='portrait', papertype=None, format=None,
    # #     transparent=False, bbox_inches='tight', pad_inches=0.1,
    # #     frameon=None, metadata=None) 
    # plt.show();
    # # plt.savefig('Anaconda3/envs/tensorflow_env/mark/satw/C_AFBR_pred_obs.svg', dpi=None, facecolor='w', edgecolor='w',
    # #      orientation='portrait', papertype=None, format=None,
    # #      transparent=False, bbox_inches='tight', pad_inches=0.1,
    # #      frameon=None, metadata=None)
    
    # print(  'X_test_mm', X_test_mm.shape,'Yhat_v', Yhat.shape) #'y_test', y_test.shape,
    ###history = AFBR.fit( X_train_mm, y_train_mm, validation_data = (X_test_mm, y_test_mm), batch_size= int(b), epochs=ep,  shuffle = False, verbose=1, callbacks=[tensorboard_callback])  #tensorboard_callback, ICANN paper: 250 epochs 
    
    #save model architecture and weights to single file
##    AFBR.save('C:/Users/mark_/mark_data/Input/AFBR_mlp_4x256.h5')
    # AFBR.save('/scratch/mmccormi1/AFBR_mlp_4x2048.h5') 
##    print("Saved model to disk")

      
   
    return Yhat, Yhat_v_mlp,  rmse_mlp#, r_value# result # #  remove when running DCR2_mlp#, test_mse Yhat_v_mlp_1_layer,
    
######### 1-layer MLP  ######

# def AFBR_1_layer_model(X_train_mm, y_train_mm, X_test_mm, y_test_mm):
#     b = 4.0  #batch_size
#     node=1
#     ep = 100 #training epochs
#     AFBR_1 =tensorflow.keras.Sequential()  # keras.
#     initializer = tensorflow.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1, seed=42) #glorot_uniform(seed=42)  RandomNormal(mean=0.0, stddev=sdv, seed=42)
#     AFBR_1.add(tensorflow.keras.Input(shape=(4,)))  #   keras.layers.Dense(4, input_shape=(4,))
#       #input_layer = input_shape=(step,dimension) #102,24
#     AFBR_1.add(layers.Dense(units=192*node, activation= 'relu', use_bias=True, kernel_initializer=initializer,
#     bias_initializer = 'ones')) #,relu
#     # AFBR_1.add(layers.Dense(units=4*node, activation= 'relu'))#, use_bias=True, kernel_initializer=initializer,
#     #bias_initializer = 'zeros'))
#     AFBR_1.add(layers.Dense(units=1)) #,, activation='linear',input_dim=(1),input_shape=(step,)  , input_shape=(24,8)units=2, , 
    
#     print(AFBR_1.summary())
    
#     # RMSprop(learning_rate =0.0001, rho=0.01, momentum = 0.3, epsilon=0.0000001, centered= True) #, epsilon= 0.9, name = "RMSprop". tf.keras.optimizers.RMSprop  Keras defaults: learning_rate =0.001, rho=0.9, momentum = 0.0, centered = False. ICANN lr=0.000000001
#     metric=tensorflow.keras.metrics.RootMeanSquaredError(name='root_mean_squared_error', dtype=None)
#     # AFBR.compile(optimizer=keras.optimizers.Adam(learning_rate=0.9, beta_1=0.9, beta_2=0.999, epsilon=0.0000001), loss="mse",  metrics = [metric]) #keras.metrics.mse()loss = 'mean_squared_logarithmic_error', "RMSprop", metrics=['accuracy'], optimizer= sgd, optimizer,'adam' RMSprop(lr=0.001  metrics = mean_squared_error
#     AFBR_1.compile(optimizer=tensorflow.keras.optimizers.RMSprop(learning_rate=0.0001, rho=0.9, momentum = 0.0, epsilon=0.0000001, centered= True), loss="mse",  metrics = [metric]) #, clipnorm=1.0
#     # AFBR_1.compile(optimizer=keras.optimizers.SGD(learning_rate=0.01,  nesterov= False), loss="mse",  metrics = [metric])#momentum=mom,
#     # Tensorboard
#     # tf.reset_default_graph()   Paste in the Spyder console
#     # g = tf.graph()
#     # Clear any logs from previous runs
#     # rm -rf \\logs\\
  
#     #log_dir = 'C:\\Users\mark_\Anaconda3\envs\Tensorflow_v2\Lib\site-packages\logs\scalars' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  ##'logs\\scalars\\' \\scalars\\  'C:\\Anaconda3\\envs\\Tensorflow_v1\\Mark\\tflearn_logs\\' Must use 2 back slashes
#     # log_dir = os.path.join('logs','fit', datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
#     # print(log_dir)
#     #tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=False, write_images=True)  #, histogram_freq=1 update_freq='epoch',
#     # tensorboard --logdir '\\Anaconda3\\envs\\Tensorflow_v2\\Mark\\tflearn_logs'    paste to the Anaconda command line

#     history = AFBR_1.fit( X_train_mm, y_train_mm, validation_data = (X_test_mm, y_test_mm), batch_size= int(b), epochs=ep,  shuffle = False, verbose=1)  # , callbacks=[tensorboard_callback]   
#     # history_t = AFBR_t.fit(X_train_mm, y_train_mm, validation_data = (X_test_mm, y_test_mm), epochs=100, batch_size= int(b), shuffle = True, verbose=1) 
#     # Get MLP training and test loss histories
        
#     # history = AFBR.fit(X_train, y_train, epochs=100, batch_size=int(step), verbose=1)
#     # with tf.compat.v1.Session() as sess:
#     _, train_mse = AFBR_1.evaluate(X_train_mm, y_train_mm, verbose=1) #batch_size= int(b),  steps=1,
#     _, test_mse = AFBR_1.evaluate(X_test_mm, y_test_mm, verbose=1) #batch_size= int(b), steps=1,sample_weight=None, steps=None, callbacks=None, batch_size=int(step)
#     print('Train RMSE: %.3f' % (train_mse))
#     print('Test RMSE: %.3f' % (test_mse))#, test_mse, Test MSE: %.1f%%
#     #print('test loss, test acc:', validate)
#     # AFBR.reset_states()
#     # tf.summary.histogram("activations", activation)
#     # _, train_mse_t = AFBR_t.evaluate(X_train_mm, y_train_mm,  verbose=1) #batch_size= int(b),  steps=1,
#     # _, test_mse_t = AFBR_t.evaluate(X_test_mm, y_test_mm,  verbose=1) #batch_size= int(b), steps=1,sample_weight=None, steps=None, callbacks=None, batch_size=int(step)
#     # print('Train-2 MSE: %.3f' % (train_mse_t))
#     # print('Test-2 MSE: %.3f' % (test_mse_t))
#  ##### Figures for information - not in MDPI article     #######################
#     plt.figure(figsize=(16,6))
#     plt.title('1-layer training loss', fontsize='large') #and testing
#     plt.plot(history.history['loss'], c="k", linestyle="--", label='training_loss') #, 'r'train_history
#     plt.plot(history.history['val_loss'], c="k", linestyle="-",label='validation test_loss') #, 'b'val_history
#     # plt.plot(history_t.history['loss'], label='training_loss-2') #, 'r'train_history
#     # plt.plot(history_t.history['val_loss'], label='test_loss-2') #, 'b'val_history
#     plt.legend()
#     plt.xlabel('Epoch')
#     plt.ylabel('Root Mean Squared Error') #Mean squared 
#     #plt.savefig('Anaconda3/envs/tensorflow_env/mark/satw/B_AFBR_loss.png', bbox_inches='tight')
#     plt.show()
#     # plt.savefig('C:/Users/mark_/Anaconda3/envs/tensorflow_env/mark/satw/B_AFBR_loss.svg', dpi=None, facecolor='w', edgecolor='w',
#     #        orientation='portrait', papertype=None, format=None,
#     #        transparent=False, bbox_inches='tight', pad_inches=0.1,
#     #        frameon=None, metadata=None)  
 
# ### Testing during model building   
#     Yhat = AFBR_1.predict([X_test_mm])#, steps=1, batch_size=4,, verbose=1
#     np.savetxt("C:/Users/mark_/Documents/02_Professional/03_Unil/SATW/MDPI_paper/MDPI_article/MDPI_article_v03/figures/Yhat_train_1-layer", Yhat, delimiter=',')
  
#     Yhat_v = AFBR_1.predict([X_test_mm])#, steps=1, batch_size=4,, verbose=1
#     np.savetxt("C:/Users/mark_/Documents/02_Professional/03_Unil/SATW/MDPI_paper/MDPI_article/Data/Yhatv_test_1-layer", Yhat_v, delimiter=',')
#     # np.savetxt("/scratch/mmccormi1/yhat_regression_loo_100.csv", Yhat_v, delimiter=',')
    
#     # Yhat_t = AFBR_t.predict([X_test_mm])#, steps=1, batch_size=4,, verbose=1
#     # print('Yhat', Yhat.shape, 'Yhat_t', Yhat_t.shape,'X_test_mm', X_test_mm.shape)
#     plt.figure(figsize=(16,6)) 
#     plt.plot((y_test_mm-Yhat), color='purple')
#     plt.xlabel('Day') #Measurement sequence number
#     plt.ylabel('Raw error (True-predicted)')
#     plt.title('1-layer model - Error during testing ', fontsize='large') #MLP predicted and observed flow rate during testing
#     plt.show();    
   
#     # mse_mlp_t = mean_squared_error(y_test_mm, Yhat_t)  #outputs,predictions
#     # print('MLP_t MSE:%.1f%%', mse_mlp_t)
#     # np.savetxt('C:/Users/mark_/Anaconda3/envs/Tensorflow_v1/Mark/mse_mlp.csv')
#     # MSE.to_csv('C:/Users/mark_/Anaconda3/envs/Tensorflow_v1/Mark/mse_mlp.csv', sep=';', index=False, decimal=',')
#     #Yhat_t = scaler.inverse_transform(X_test_mm) #Yhat.reshape (-1,1)
        
#     #Graph predicted and observed
#     plt.figure(figsize=(16,6)) 
#     #plt.plot(Yhat_t[:,0], 'r')
#     #plt.plot(Yhat_t[:,1], 'c')
#     #plt.plot(Yhat_t[:,2], 'g')
#     #plt.plot(Yhat_t[:,3])#, 'y'
#     plt.plot(y_test_mm,c="g", linestyle="-", label = 'True outputs (Test data)')#, 'y'
#     plt.plot(Yhat, c="k", linestyle="-", label ='Predictions (Test data)' )#, 'y'[:,0]
#     # plt.plot(Yhat_t, label= 'Predictions (Test-2 data)') #, 'y'[:,0]
#     #plt.plot(y_test)#, 'y'
#     plt.legend()#['True outputs (Test data)', 'Predictions (Test data)', 'Predictions (Test-2 data)'])
#     plt.xlabel('Day') #Measurement sequence number
#     plt.ylabel('Daily CH4 production (MinMax scaled)')
#     plt.title('1-layer model - accuracy during testing ', fontsize='large') #MLP predicted and observed flow rate during testing
#     # plt.savefig('C:/Users/mark_/Documents/02_Professional/03_Unil/SATW/MDPI_paper/MDPI_article/MDPI_article_v03/figures/F4_mlp_1-layer.svg', format="svg")
#     plt.show();
    
#     mse_mlp = mean_squared_error(y_test_mm, Yhat)  #outputs,predictions
#     print('1-layer MLP MSE is :%.1f%%', mse_mlp)
    
#     rms = sqrt(mean_squared_error(y_test_mm, Yhat))
#     print("rms error is: " + str(rms))
    
#     print(  'X_test_mm shape', X_test_mm.shape,'Yhat shape', Yhat.shape) #'y_test', y_test.shape,
    
    
#     # save 1-layer perceptron model architecture and weights to single file
#     # AFBR_1.save('C:/Users/mark_/Anaconda3/envs/Tensorflow_v2/Mark/AFBR_1-layer.h5')
#     # print("Saved model to disk")
#     return Yhat, Yhat_v #Yhat, train_mse, AFBR#, test_mse 

####  Keras LSTM    #########
# In the console type numpy.__version__ to get the version
###  Reference: https://machinelearningmastery.com/use-different-batch-sizes-training-predicting-python-keras/

def afbr_lstm(X_data, y_data):   #inputs_train, outputs_train, inputs_test, outputs_test
    #inputs: A 3D tensor with the default batch_major shape [batch (or number of samples), timesteps, number of features]. Number of samples = length of vector, timesteps = 1 (day), number of features = 4 predictors 
    # Batch = input length (sample size)/number of time steps.
    # You can set RNN layers to be 'stateful', which means that the states computed for the samples in one batch will be reused as initial states for the samples in the next batch.
    ep = 100  #number of epochs
    
    b = 170 #170# 153*2=306# batch size in days. The batch size limits the number of samples to be shown to the network before a weight update can be performed. Use 170 because 170 samples.  1530*0.2 = 306. 306+1224=1530. 306/9=9. 1224/34=36
    t = 1 # number of time steps. 170/17 = 10 days.  Creates the length of the recurrent cell window, in days. Sample size must be divisable by time step.  170/17 = 10 days
    f = 4 # number of features
####
##    X_data_mm = []
##        # apply min-max scaling
##    for column in range(4): #df_norm.columns
##        X_data_mm.append([(X_data[:,column] - X_data[:,column].min()) / (X_data[:,column].max() - X_data[:,column].min())])
##    
##    X_data_mm = np.vstack(X_data_mm).T
##
##    #  y_data Yeo-Johnson transformed
##    y_data_yj = sklearn.preprocessing.power_transform(y_data.reshape(-1,1), method='yeo-johnson', standardize=True, copy=True)  # box-cox
##    y_data_yj_mm = (y_data_yj[:] - y_data_yj[:].min()) / (y_data_yj[:].max() - y_data_yj[:].min())
##    Y_data_mm = y_data_yj_mm
##    # y_data_yj_mm = y_data_yj_mm.reshape((-1,9), order='F') 
    
    scale_tanh = MinMaxScaler(feature_range=(-1,1))
    X_data_mm = scale_tanh.fit_transform(X_data)
    y_data_mm = scale_tanh.fit_transform(y_data)
    print(X_data.shape)
    print(y_data.shape)
    print(X_data_mm.shape)
    print(y_data_mm.shape)
    X_data_mm = np.reshape(X_data_mm, (-1, t, f))
    y_data_mm = np.reshape(y_data_mm,(-1, 1))
    inputs_train, inputs_test, outputs_train, outputs_test = train_test_split(X_data_mm, y_data_mm, test_size=340 , random_state= 42, shuffle = True, stratify= None)  # 306 = 9*34. 340=2*170

    # OR Use if 2d array needs to be converted to 3d array
##    inputs_train_s, outputs_train_s, inputs_test_s, outputs_test_s = train_test_split(X_data_mm, y_data_mm, test_size=306 , random_state= 42, shuffle = True, stratify= None) #X_train_mm, X_test_mm, y_train_mm, y_test_mm 
##    inputs_train = np.reshape(inputs_train_s, (-1, t, f))   # r = reshaped for lstm. Batch-major (default) form = [batch, timestep, feature]. This implies that time_major = False.  (inputs_train_s, (-1, inputs_train_s.shape[1], t)) 
##    inputs_test = np.reshape(inputs_test_s, (-1, t, f))  #inputs_train.shape[0]
##    outputs_train = np.reshape(outputs_train_s,(-1, 1) )   # r = reshaped for lstm. Batch-major (default) form = [batch, timestep, feature]. This implies that time_major = False.  (-1, t, outputs_train_s.shape[1])
##    outputs_test = np.reshape(outputs_test_s,(-1, 1) )  #inputs_train.shape[0]

    #return inputs_train, outputs_train, inputs_test, outputs_test

    ## Use if dataframe needs to be converted to array
    # inputs_train_r = np.reshape(inputs_train.to_numpy().astype('float32'), (inputs_train.shape[0], 1, inputs_train.shape[1]))   # r = reshaped for lstm
    # inputs_test_r = np.reshape(inputs_test.to_numpy().astype('float32'), (inputs_test.shape[0], 1, inputs_test.shape[1])) 
    
    # outputs_train_r = np.reshape(outputs_train, (outputs_train.shape[0], 1, outputs_train.shape[1]))
    # outputs_test_r = np.reshape(outputs_test, (outputs_test.shape[0], 1, outputs_test.shape[1]))
    # # a = inputs_train.to_numpy().astype('float32')
    # # a = np.asarray(inputs_train).astype('float32')
    # # z = tf.constant(a)
    # inputs_train_r =  inputs_train.to_numpy().astype('float32').reshape(inputs_train.to_numpy().astype('float32').shape[0], 1, inputs_train.to_numpy().astype('float32').shape[1])#tf.convert_to_tensor(a)##np.reshape(inputs_train, (inputs_train[:,0], 1, inputs_train[:,1]))   # r = reshaped for lstm
   
    # inputs_test_r = inputs_test.to_numpy().astype('float32').reshape(inputs_test.to_numpy().astype('float32').shape[0], 1,inputs_test.to_numpy().astype('float32').shape[1])#tf.convert_to_tensor(inputs_test.to_numpy().astype('float32'))#np.reshape(inputs_test, (inputs_test.shape[0], 1, inputs_test.shape[1]))
    #outputs_train_r = outputs_train#.to_numpy().astype('float32')#tf.convert_to_tensor(outputs_train)#np.reshape(outputs_train, (outputs_train.shape[0], 1, outputs_train.shape[1]))
    #outputs_test_r = outputs_test#.to_numpy().astype('float32')#tf.convert_to_tensor(outputs_test) #np.reshape(outputs_test, (outputs_test.shape[0], 1, outputs_test.shape[1]))     
    # a = pd.DataFrame(inputs_train).to_numpy()
    
    ## return_sequences: Default=False. True => return only the last output. False => returns the full sequence. For layers, set to True.
    ## stateful: default = False. If True, the last state for each sample at index i in a batch will be used as initial state for the sample of index i in the following batch.

    ## Version 1 (works in some cases...when t=1)
    lstm_model = keras.Sequential()
##    lstm_model.add(layers.Input(shape=(b, 4))) #shape=(None, b, 4)
##    lstm_model.add(layers.Embedding(input_dim=1000, output_dim=64))
    lstm_model.add(layers.LSTM(128, activation="tanh", recurrent_activation="sigmoid", batch_input_shape=(b,t,f), return_sequences=False, stateful=True, time_major=False))   #batch_input_shape=(b,t,f) if stateful=True, input_shape=(t,f),
    ### lstm_model.add(layers.LSTM(128, activation="tanh", recurrent_activation="sigmoid", input_shape=(t,f), return_sequences=True, stateful=False))
    ### lstm_model.add(layers.LSTM(128, activation="tanh", recurrent_activation="sigmoid", input_shape=(t,f), return_sequences=True, stateful=False))
    ### lstm_model.add(layers.LSTM(128, activation="tanh", input_shape=(t,f), return_sequences=True))
    lstm_model.add(layers.Dense(1))
##    x = tf.constant([0 for i in range(128)], dtype = tf.float32)
##    tf.keras.activations.relu(x, alpha=0.5, max_value=5.0, threshold=0.05)
##
##    ## Version 2 API?  https://machinelearningmastery.com/use-different-batch-sizes-training-predicting-python-keras/
##    input_layer = Input(shape=(t, f))  # When stateful = True, provide batch_input_shape=(b,t,f) , 
##    lstm=LSTM(128, activation="relu", recurrent_activation="sigmoid", return_sequences=False, stateful=False, time_major=False)(input_layer)   #tanh, batch_input_shape=(b,t,f) if stateful=True, input_shape=(t,f). time_major = False =>[batch, timesteps, feature]
##    #lstm=LSTM(128, activation="tanh", recurrent_activation="sigmoid", return_sequences=False, stateful=True)(lstm)
##    # lstm_model.add(layers.LSTM(128, activation="tanh", recurrent_activation="sigmoid", input_shape=(t,f), return_sequences=True, stateful=False))
##    # lstm_model.add(layers.LSTM(128, activation="tanh", input_shape=(t,f), return_sequences=True))
##    dense=Dense(1)(lstm)
##    lstm_model=Model(inputs= input_layer, outputs=dense)

    lstm_model.summary()
    lstm_model.compile( optimizer='adam', loss=tensorflow.keras.losses.MeanSquaredError()) #loss='mean_squared_error',
##    lstm_model.fit(inputs_train_r, outputs_train_r, validation_data = (inputs_test_r, outputs_test_r), epochs=50, batch_size=b, verbose=2, shuffle=False)
##    history = lstm_model.fit(inputs_train_r, outputs_train_r, validation_data = (inputs_test_r, outputs_test_r), epochs=50, batch_size=b, verbose=2)
    history = lstm_model.fit(inputs_train, outputs_train, validation_data = (inputs_test, outputs_test), batch_size=b, epochs=ep, verbose=0, shuffle=False) #batch_size=b, , steps_per_epoch=None

    #test_scores = lstm_model.evaluate(inputs_test, outputs_test, batch_size=b, verbose=0)
    
    #Yhat = lstm_model.predict(inputs_train_r, batch_size=1).flatten()  # Yhat is predictions on training data. Batch_size of the predictions is different thant batch_size used for training.
    #Yhat_v_l = lstm_model.predict(inputs_test_r, batch_size=1).flatten() # Yhat_v is predictions on testing data (validation). Batch_size of the predictions is different thant batch_size used for training.
    #Batch_size of the predictions is different thant batch_size used for training., batch_size=bs
    #t=1
    Yhat = lstm_model.predict(inputs_train, batch_size=b).flatten()  #.reshape(-1,t,4) Yhat is predictions on training data. Batch_size of the predictions is different thant batch_size used for training., batch_size=1
    Yhat_v_lstm = lstm_model.predict(inputs_test, batch_size=b).flatten() #.reshape(-1,t,4) Yhat_v is predictions on testing data (validation). Batch_size of the predictions is different thant batch_size used for training., batch_size=1
   
    #history = AFBR.fit(X_train_mm, y_train_mm, validation_data = (X_test_mm, y_test_mm), batch_size= int(b), epochs=ep,  shuffle = False, verbose=0)  # , inputs_test, outputs_test,  shuffle = True to shuffle before splitting., verbose=3, callbacks=[tensorboard_callback], tensorboard_callback, ICANN paper: 250 epochs    
    #history_t = AFBR_t.fit(X_train_mm, y_train_mm, validation_data = (X_test_mm, y_test_mm), epochs=100, batch_size= int(b), shuffle = True, verbose=1) 
    #history = AFBR.fit(X_train, y_train, epochs=100, batch_size=int(step), verbose=1)
    # test_scores = AFBR.evaluate(inputs_test_r, outputs_test_r, verbose=0) #X_test_mm, y_test_mm
    
    # print("Test loss:%.1f%%", history.history['loss'])#[0]
    # print("Test accuracy:", history.history['val_loss'])
   
    # print("Mean absolute error:", test_scores[0])
    # print( history.history.keys())
    # test_mse = test_scores[0]
    # print('input shape:%.1f%%', inputs)
    # print('input test shape:%.1f%%', inputs_test)
    
   
    ######### LSTM loss        ###########
    
##    plt.figure(figsize=(10,15))
##    # plt.title('MLP loss', fontsize='large') #and testing
##    plt.plot(history.history['loss'], c="r", linestyle="-", linewidth = 4, label='training_loss') #, 'r'train_history
##    plt.plot(history.history['val_loss'], c="b", linestyle="-", linewidth = 2, label='test_loss') #, 'b'val_history, val_loss
##    plt.legend()
##    plt.ylim(0.00, 0.125)
##    plt.xlabel('Epoch')
##    plt.ylabel('Mean Squared Error') 
##    plt.title('LSTM model - Loss during training and testing', fontsize='large')
##    plt.savefig('C:/Users/mark_/Documents/02_Professional/03_Unil/SATW/MDPI_paper/complementary_information/figures/Fig_lstm_loss.svg', format="svg")
##    plt.show()
     
    outputs_test_it =scale_tanh.inverse_transform(outputs_test[:,0].reshape(-1, 1))
    Yhat_v_l_it = scale_tanh.inverse_transform(Yhat_v_lstm.reshape(-1, 1))  # _it =  inverse transform of the predictions

    y2 = np.array([0, 200])#len(outputs_test_it)]) 
    x2 = np.array([0, 200])#len(outputs_test_it)])
    
##    plt.figure()#figsize=(12,12)) 
##    #labels=["True values", "Predictions"]
##    plt.ylim(0, 200)
##    plt.xlim(0,200)
##    #plt.plot(y_test_mm)
##    #plt.plot(Yhat_v_l_it)
##    #plt.plot(Yhat_v_l)
##    #plt.plot(outputs_test)
##    #plt.scatter(outputs_test_r, Yhat_v_l). This gives an array with 306 lines : outputs_test.reshape(-1,1)
##    plt.scatter(outputs_test_it, Yhat_v_l_it)  #This works: outputs_test[:,0], Yhat_v_l
##    plt.plot(x2,y2, c='k')
##    plt.title('LSTM model - Predicted and True values, testing ', fontsize='large') #MLP predicted and observed flow rate during testing
##    plt.xlabel('True values')
##    plt.ylabel('Predictions') #Mean squared
##    plt.savefig('C:/Users/mark_/Documents/02_Professional/03_Unil/SATW/MDPI_paper/complementary_information/figures/Fig_lstm_T_vs_P.svg', format="svg")
##    plt.show()

##    plt.figure(figsize=(16,6)) 
##    labels=["True values", "Predictions"]
##    plt.plot(outputs_test_it)
##    plt.plot(Yhat_v_l_it)
##    plt.title('LSTM model - Predicted and True values, testing ', fontsize='large') #MLP predicted and observed flow rate during testing
##    plt.legend(labels)
##    plt.savefig('C:/Users/mark_/Documents/02_Professional/03_Unil/SATW/MDPI_paper/complementary_information/figures/Fig_lstm_T_and_P_time.svg', format="svg")
##    plt.show() 
##    
    
     #save model architecture and weights to single file
    # lstm_model.save('C:/Users/mark_/Anaconda3/envs/Tensorflow_v2/Mark/lstm_model_1x128.h5')

    scale_MM = MinMaxScaler(feature_range=(0,1))
    Yhat_v_lstm = scale_MM.fit_transform(Yhat_v_lstm.reshape(-1, 1))
    outputs_test = scale_MM.fit_transform(outputs_test.reshape(-1, 1))
    
    return Yhat, Yhat_v_lstm, inputs_test, outputs_test, scale_tanh #, outputs_test_r


    # from math import sqrt
    # testScore = sqrt(mean_squared_error(outputs_test[:,0], testPredict[:,0]))
    # print('Test Score: %.2f RMSE' % (testScore))
    
    # MAE = mean_absolute_error(outputs_test[:,0], testPredict[:,0], sample_weight=None, multioutput='uniform_average')
    # print('Mean absolute error: %.3f MAE' % (MAE))
     
    # time = np.arange(1, 460, 1) #(x for x in len(outputs_test))
    
    # plt.plot(testPredict[:,0], outputs_test[:,0])
    # plt.plot(testPredict[:,0], outputs_test[:,0])
    # plt.scatter(time, testPredict[:,0])
    # plt.scatter(time, outputs_test[:,0])

###################   Evaluation of the experiental plan Leave one column out   #####################  
#########  Use only during dropped column runs to evaluate experimental plan   ########
def NN_DC(y_data_mm, X_data_mm):   #  DC = Delete Column. .to_numpy(dtype='float32')
    from satw_data_p import NN_preprocess
#     y_data_dc = [] 
    X_data_mm_dc = pd.DataFrame()  
#     X_data_mm = []
#     X_data_sel = []
#     Yhat = []  
#     Yhat_v = []
#     X_train_mm = []
#     X_test_mm = []
#     y_train_mm = []
#     y_test_mm = []
# # # # X_data_mm, y_data_mm = NN_LX_traindata()
# y_data log transformed
# y_data_t = np.log(y_data)  #log1p
    # y_data_t_mm = (y_data_t[:] - y_data_t[:].min()) / (y_data_t[:].max() - y_data_t[:].min())
    # y_data_in = np.reshape(y_data_t_mm, (170,-1),  order='F')
    
    
    # X_data_mm.columns = ['ESD', 'MAT', 'HDR', 'Qin']  #X_data
    #X_data_mm_dc = np.split(X_data_mm.to_numpy(copy=False), 9) #X_data
##    print('X_data_mm shape before split',np.asarray(X_data_mm).shape)
##    print('X_data_mm head before split',X_data_mm[:10])
    X_data_mm_dc = np.split(X_data_mm, 9) #X_data
##    print('X_data_mm_dc shape after split',np.asarray(X_data_mm_dc).shape)
##    print('X_data_mm_dc head after split',X_data_mm_dc[:10])
    ES1 = X_data_mm_dc[0], 
    ES2 = X_data_mm_dc[1], 
    ES3 = X_data_mm_dc[2], 
    ES4 = X_data_mm_dc[3], 
    ES5 = X_data_mm_dc[4] , 
    ES6 = X_data_mm_dc[5], 
    ES7 = X_data_mm_dc[6], 
    ES8 = X_data_mm_dc[7] , 
    ES9 = X_data_mm_dc[8] 
    
    # cols = ['ES1', 'ES2', 'ES3', 'ES4', 'ES5', 'ES6', 'ES7', 'ES8', 'ES9']
    # DataFrame.to_dict(orient='dict', into=<class 'dict'>)
    
    L9 = {0:ES1, 1:ES2, 2:ES3, 3:ES4, 4:ES5, 5:ES6, 6:ES7, 7:ES8, 8:ES9} 
    keys= np.arange(0,9)
##    print('y_data_mm head before split',y_data_mm[:10])
##    print('y_data_mm shape before split',np.shape(y_data_mm))
    rdel = np.random.randint(0, high=9)#len(y_data_mm)+1 rdel = random selection of the column to be deleted
    y_data_in = np.reshape(y_data_mm.to_numpy(), (170,-1),  order='F')  # Checked and confirmed that F order is required to preserve order
##    print('y_data_mm head before split',y_data_mm[:10])
    print('y_data_in shape after reshape',np.shape(y_data_in))
    y_data_mm_dc = np.delete(y_data_in, rdel, axis=1).flatten()#.reshape(-1,1).values() # New array with loo column selected at random
    #y_data_mm_dc.flatten()
    # index = np.array(np.delete(keys, rdel, axis=None))
    # X_loo = [L9[x+1] for x in index] #  Delete the random selection of a redundant experiment from predictors
    #del L9[rdel]
    del X_data_mm_dc[rdel]  ##Delete the experiment from the array
    X_data_mm_dc = np.concatenate(X_data_mm_dc, axis=0)

##    print('y_data_mm_dc final shape', np.asarray(y_data_mm_dc).shape)
##    print('y_data_mm_dc head final',y_data_mm_dc[:10])
##    print('X_data_mm_dc final shape',np.asarray(X_data_mm_dc).shape) 
##    print('X_data_mm_dc head final',X_data_mm_dc[:10])


##    plt.figure()
##    plt.plot(y_data_mm_dc)
##    #plt.plot(y_data_mm)
##    plt.title('y data dc')
##    plt.show()
##
##    plt.figure()
##    #plt.plot(y_data_mm_dc)
##    plt.plot(y_data_mm)
##    plt.title('y data all')
##    plt.show()
##
##    plt.figure()
##    plt.plot(X_data_mm_dc)
##    #plt.plot(X_data_mm)
##    plt.title('X data dc')
##    plt.show()
##
##    plt.figure()
##    #plt.plot(X_data_mm_dc)
##    plt.plot(X_data_mm)
##    plt.title('X data all')
##    plt.show()

    ####  Select the response data and Min-Max scale the predictor data after having removed 1 experiment
    # y_data_mm = y_data_mm_dc 
    
    # X_data_mm = np.empty([])    
    # X_data_sel = X_data_mm_dc[:,0:4]  #Select X_data for min-max scaling. X_data.copy()
    
    # scaler = MinMaxScaler()
    # print(scaler.fit(X_data_sel))
    # MinMaxScaler()
    # print(scaler.data_max_)
    # print(scaler.data_min_)
    # X_data_mm = scaler.fit_transform(X_data_sel, y=None)
    #########  Use only during runs with 1 experiment removed
    # X_data_mm = X_data_mm_dc
    # y_data_mm = y_data_mm_dc
    # X_train_mm, X_test_mm, y_train_mm, y_test_mm = NN_preprocess(X_data_mm_dc, y_data_mm_dc) 
    X_train_mm_dc, X_test_mm_dc, y_train_mm_dc, y_test_mm_dc = train_test_split(X_data_mm_dc, y_data_mm_dc, test_size=0.2, random_state= 42, shuffle = True, stratify= None)  #shuffle = True, random_state=42, stratify= None
    Yhat, Yhat_v_mlp, _ = AFBR_MLP_model(X_train_mm_dc, y_train_mm_dc, X_test_mm_dc, y_test_mm_dc)  #  . or l-layer model. X_train_mm, y_train_mm, X_test_mm, y_test_mm, inputs, outputs, inputs_test, outputs_test
    Yhat_dc, Yhat_v_dc = Yhat, Yhat_v_mlp
#### Remove rmse_mlp, result from AFBR_MLP_model return line   
    ########  Use only during loo runs
    # y_data_dc = [] 
    # X_data_dc = pd.DataFrame()  # Restore original y_data_mm
    # # X_data_mm = []
    # X_data_sel = []
    # Yhat = []  
    # Yhat_v = []
    # X_train_mm = []
    # X_test_mm = []
    # y_train_mm = []
    # y_test_mm = []

    # Yhat_train_all= np.concatenate([np.array(j) for j in Yhat_rep], axis = 1) #Yhat_rep # axis = 0 -> by rows
    # Yhat_test_all_dc= np.concatenate([np.array(j) for j in Yhat_v_rep], axis = 1) #Yhat_v_rep = predicted values from validation runs
    
    # R2_dc = [] 
    # Rsq_dc = []  
    # rms_mlp_dc = []
    # for r in range(Yhat_test_all_dc.shape[1]):   
    #     reg_dc = LinearRegression(fit_intercept=False).fit(y_test_mm.reshape(-1,1), Yhat_test_all_dc[:,r]) #.reshape(-1,1), fit_intercept=False forces intercept to zero, Yhat_polynomial
    #     R2_dc.append(reg_dc.score(y_test_mm.reshape(-1,1), Yhat_test_all_dc[:,r]))
    #     Rsq_dc.append(scipy.stats.linregress(y_test_mm, Yhat_test_all_dc[:,r]).rvalue)
    #     rms_mlp_dc.append(sqrt(mean_squared_error(y_test_mm, Yhat_test_all_dc[:,r])))
    
    # rms_mlp_dc_mean = np.mean(rms_mlp_dc)
    # Rsq_dc_mean = np.mean(Rsq_dc)
    # Yhat_v_dc_mean = np.mean(Yhat_test_all_dc, axis =1)
    # y_test_mm_dc = y_test_mm
    
    # print("rms_mlp_dc_mean", rms_mlp_dc_mean)
    # print("Rsq_dc_mean", Rsq_dc_mean)
    # print("Yhat_v_dc_mean", Yhat_v_dc_mean)
     
    # np.savetxt("/scratch/mmccormi1/rms_mlp_dc_mean.csv", rms_mlp_dc_mean, delimiter=";", fmt="%10.2f") 
    # np.savetxt("/scratch/mmccormi1/Rsq_dc_mean.csv", Rsq_dc_mean, delimiter=";", fmt="%10.2f") 
    # np.savetxt("/scratch/mmccormi1/Yhat_v_dc_mean.csv", Yhat_v_dc_mean, delimiter=";", fmt="%10.2f") 
    # np.savetxt("/scratch/mmccormi1/y_test_mm_dc.csv", y_test_mm_dc, delimiter=";", fmt="%10.2f")
    
    return Yhat_dc, Yhat_v_dc, y_test_mm_dc #rms_mlp_dc_mean, Rsq_dc_mean, Yhat_v_dc_mean    

    
####### SIMULATION prediction ##### 
def AFBR_MLP_sim(SX_test_mm):#, AFBR_s):  #, , X_train_mm, y_train_mm, X_test_mm, y_test_mm, SX_test_mm_r, mSX_test_mm
## load model if not runing multiple simulations
   
    # AFBR_s = load_model('C:/Users/mark_/Anaconda3/envs/Tensorflow_v2/Mark/AFBR_mlp_4x2048.h5')
    # AFBR_s = load_model('C:/Users/mark_/Anaconda3/envs/Tensorflow_v2/Mark/AFBR_mlp_v2-3-layer.h5')
    # AFBR_s = load_model('C:/Users/mark_/Anaconda3/envs/Tensorflow_v2/Mark/AFBR_mlp_v2-6-layer-do1.h5')
    # AFBR_s = load_model('C:/Users/mark_/Anaconda3/envs/Tensorflow_v2/Mark/AFBR_mlp_v3.h5')
    AFBR_s = load_model('C:/Users/mark_/mark_data/Input/AFBR_mlp_4x256.h5')
    # # summarize model.
    AFBR_s.summary()
    Yhat_s = []
    Yhat_s = AFBR_s.predict([SX_test_mm])#.reshape(-1,1)#Yhat_s_r = AFBR_s.predict([SX_test_mm_r]

    #print('Yhat_s', Yhat_s)
    print('Yhat_s - length of prediction for 1 simulation run', len(Yhat_s))
    #SYhat_t = scaler.inverse_transform(SX_test_mm)
    # print('Yhat_s shape', Yhat_s.shape, 'SX_test_mm',SX_test_mm.shape) # 'SYhat_t', SYhat_t.shape
    # plt.figure(figsize=(16,6)) 
    # # Yhat_s = pd.DataFrame(Yhat_s)
    # plt.plot(Yhat_s, c='b', label="Predictions (Test data)")# [:]marker='+', linestyle='None',
    # plt.xlabel('Day')
    # plt.ylabel('Daily CV reduction (MinMax scaled)')
    # plt.title("Simulation MLP NN (Predicted CV reduction)", fontsize='large')
    # plt.legend()
    # plt.show()
    
##    S =[] # the sample of length sl that is sliced from the predictions (Yhat_s)
##    
##    # Scumi =[] # the sample of length sl that is sliced from the predictions (Yhat_s)
##    # Scum = pd.DataFrame(Scumi)
##    
##    p=125  # the number of permutations of the predictors (27 from the L9 plan). 125 from the 3^5 plan. 64 from 4 level, 3 factors
##    #l=0
##    
##    sn = int(np.divide(len(Yhat_s),p))# number of samples
##    # print(c)
##    
##    S = Yhat_s.reshape((p,sn))
##    # T = S[:,2].sum()
##    # print(T)
##    print('S',S[20,:])
##    Scum =S.sum(axis=1)  #  For each experiment, calculate the sum of the predictions.
    Scum = Yhat_s # #np.sum(Yhat_s)
    
    print('Scum - sum of prediction for 1 simulation run', Scum)
    #print('Scum or Yhat, Yhat_v_mlp, rmse_mlp - length of prediction for 1 simulation run', len(Scum))
    # Scum.to_csv('C:/Users/mark_/Anaconda3/envs/Tensorflow_v1/Mark/bg_sum_mlp.csv', sep=';', index=False, decimal=',')
##    Scumrank = Scum.rank(axis=0, ascending=True)
##    print('Yhat_s rank after each simulation',Scumrank)
    # Scumrank.to_csv('C:/Users/mark_/Anaconda3/envs/Tensorflow_v1/Mark/rank_mlp.csv', sep=';', index=False, decimal=',')
    
    ###### Use this code to make a graph of a single simulation run
    # Q1 =np.arange(1,28,1)  # Experiment number
    # print('Sample', len(S), 'Q1', len(Q1), 'Sum', len(Scum))
   
    # plt.figure(figsize=(16,6)) 
    # plt.scatter(Q1, Scum) #t[i][0:p]
    # plt.grid(b=True, which='major', axis='both')
    # plt.tick_params(axis='x', which ='both')
    # plt.xticks(np.arange(1, 28, step=1))
    # plt.xlabel('L9 Permutation number')
    # plt.ylabel('Cumulative daily biogas flow (Min-Max scaled)')
    # plt.title('AFBR - Simulation using the MLP model - Predicted biogas production', fontsize='large')
    # plt.show()
    # c=0
    
    return Scum  #Yhat, Yhat_v_mlp, rmse_mlp  
       

    
