# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 21:43:14 2020

@author: mark_
"""

####  This file contains code for:
    # MDPI figure 1 (HRT and CV reduction in time)
    # MDPI figure 2 (Reference and L4 time series CV reduction data)
    # Importing experimental data and surrogate from Excel files
    # Making plots that describe the bioreactor performance (used in the SATW intermediate report)
    # Creating the L4 plan data
    # Preprocessing for NN development (training and test split)
    # Generation of data used in multiple simulations using the MLP model
    # Calculating the KS statistic, see line 59
    
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split 
import matplotlib.pyplot as plt

from numpy.random import default_rng

import matplotlib.pyplot as plt
# from matplotlib.sankey import Sankey
# import Sankey
import itertools
import datetime
#import matplotlib.dates as mdates
import tensorflow as tf
# import sys
# import random
import csv
import scipy
from scipy.optimize import curve_fit
from math import sqrt
from scipy.stats import ks_2samp, shapiro, kurtosis, skew, zscore
from sklearn.metrics import mean_squared_error, r2_score
# from sklearn.model_selection import GroupKFold
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler, StandardScaler, QuantileTransformer, Normalizer, power_transform#, RobustScaler, MaxAbsScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from matplotlib import cm
# from sklearn.cross_decomposition import PLSRegression
# from random import seed
# import glob
# import os

###### Sankey test ########
# plt.figure()#figsize=(20,10)
# sankey = Sankey(scale=0.01, unit="%", gap=1.75, offset=0.8, margin=1)
# sankey.add(flows=[100, -2, -3, -9, -8, -4, -16, -58], 
#            labels=['Influent', 'final effluent', 'Building heating (CHP)', 'Digester heating (CHP)', 'Electricity (CHP)', 'Waste heat (CHP)', 'Waste sludge', 'Aerobic metabolism'],
#            pathlengths= [1, 0.25, 0.2, 0.7, 1.1, 0.7, 0.2, 0.6],#[0.01, 0.4, 0.5, 0.05, 0.02, 0.05, 0.05, 0.05],
#             trunklength=2.5,
#            orientations=[0, 0, -1,-1, -1, 1, 1, 1])  # orientation of 1 = comes from side
# sankey.finish()
# # plt.title('Experimental results', fontsize='large')
# plt.savefig('C:/Users/mark_/Documents/01_Personal/Admin/website/step_sankey.svg', format="svg")
# plt.show()


########   DO NOT USE WHEN WORKING WITH HPC CLUSTERS    #############

####  Evaluate effect of the number of nodes, layers and the sample size   #######
def NN_architecture():
    # from satw_mlp_v5 import AFBR_MLP_archopti
    # X_train_mm = pd.read_csv('C:/Users/mark_/userdata/Output/X_train_mm.csv', sep=',', decimal='.', header=None, index_col=False)
    # y_train_mm = pd.read_csv('C:/Users/mark_/userdata/Output/y_train_mm.csv', sep=',', decimal='.', header=None, index_col=False)
    # X_test_mm = pd.read_csv('C:/Users/mark_/userdata/Output/X_test_mm.csv', sep=',', decimal='.', header=None, index_col=False)
    # y_test_mm = pd.read_csv('C:/Users/mark_/userdata/Output/y_test_mm.csv', sep=',', decimal='.', header=None, index_col=False)
    
    # hl = 3 # set to the number of hidden layers
    # u = 128  # set to the number of units (nodes)
    
    # Sam_opti = [] # array of the results from which the optimized number of samples will be selected
    # step = [1, 1.33, 1.66, 2, 2.5, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    # # test = []
    # # s=34  # number of samples to select for model building
    # # sel = np.arange(0,171,s)
    # # test = np.array(X_train_mm)[np.ix_(sel,[0,1,2,3])]
    # for s in step: 
    #     sel =[int(x) for x in np.arange(0,171,s)]
    #     X_train_ = np.array(X_train_mm)[np.ix_(sel,[0,1,2,3])]
    #     y_train_ = np.array(y_train_mm)[np.ix_(sel,[0])]
    #     X_test_ = np.array(X_test_mm)[np.ix_(sel,[0,1,2,3])]
    #     y_test_ = np.array(y_test_mm)[np.ix_(sel,[0])]
    #     Yhat_o, Yhat_v_o, test_mse, mse_eval = AFBR_MLP_archopti(X_train_, y_train_, X_test_, y_test_, hl, u)  #inputs, outputs, inputs_test, outputs_test
    #     Sam_opti.append(mse_eval)#.reshape(-1,4))
    #     sel = []
    #     # print(sel) 
    # # np.savetxt("C:/Users/mark_/userdata/Output/Sam_opti.csv", Sam_opti , delimiter=",", fmt="%10.4f")
    
    # xa = []    
    # for s in step:
    #     xa.append(np.divide(170, s)) 
    
    # plt.figure() 
    # # plt.xticks(str(np.array[170:0:5])) #np.divide(170, step
    # plt.scatter(xa, Sam_opti)#, marker = "+")
    # # plt.plot(Sam_opti, marker = "+", linestyle=" ")
    # plt.ylabel("MSE")
    # plt.xlabel("Number of samples")
    # plt.title("Effect of the number of samples on the MSE ")
    # # plt.savefig('C:/Users/mark_/Documents/02_Professional/03_Unil/SATW/MDPI_paper/MDPI_article/MDPI_article_v19/figures/MLP_samplesize.svg', format="svg")     
    # plt.show()
    
#######  ARCHITECTURE OPTIMIZATION      ###############
    
    # ns = number of samples per simulated experiment = 170.  ns, Qr
    # Qr =experimental influent flow profile
    ##### Generate a list of all permutations of the mechanical set point values and the influent flow profile
    #import itertools. Number of permutations = levels ^ factors
        #perm = list(itertools.product([4,12,36],[1,2,3],[0.5,1.8,4]))
        # perm = list(itertools.product([4,12,36],[1,2.4,12],[0.5,1.8,4])) # sphere diameter, Material activity, H/D
        # print(perm)
        # NN_arch = list(itertools.product([2,3,4,5],[16,32,64,128,256, 512, 1024, 2048, 4096])) # sphere diameter, Material activity, H/D
        # print(NN_arch[1][1])
        # print(len(NN_arch))
            
        # #### Create a long sample array of mechanical properties for use as predictor variables in the SIMULATION
        # #length of sample data = 187-9 = 179 lines of data
        # A=pd.DataFrame() #columns = [0,1,2]
        # mSX_test_T=pd.DataFrame() #columns = [0,1,2]. 170 x 125 = 22125 lines
        # permar = pd.DataFrame(perm_5)
        # #print(permar.transpose())
       
        # while j < (len(NN_arch)):   
        #     for i in range(n):  
        #         A = permar.iloc[j,0:3]  # A is the data frame of the permuted mechanical values used in each sample.
        #     #print(T.loc[i])
        #         Arch_test = pd.concat([mSX_test_T,A],axis=1,ignore_index=True)
        #         #i += 1
        #     j += 1
        # mSX_test = mSX_test_T.T  # matrix of mechanical predictors (does not include influent flow rate)
        # print(mSX_test.shape)
        
    # layer = [1,2,3,4,5]#]#]#NN_arch[]
    # node=  [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]#]#]
    # Arch_NN = []
    # Archopti_test = []
    # Archopti_test_mse = []
    # Archopti_mse_eval = []
    # # h=0#[[0]]
    # # n=0
    # for h in layer:
    #     # print(h)
    #     for n in node:
    #         # print(n)
    #         # Yhat_o, Yhat_v_o, test_mse, mse_eval = AFBR_MLP_archopti(X_train_mm.to_numpy(dtype=float), y_train_mm, X_test_mm.to_numpy(dtype=float), y_test_mm, h, n)  #inputs, outputs, inputs_test, outputs_test
    #         Arch_NN.append([h,n])
    #         # Archopti_test.append(Yhat_v_o)
    #         # Archopti_test_mse.append(test_mse)
    #         # Archopti_mse_eval.append(mse_eval)
    #         # n+=1
    
    # Arch_NN = np.hstack(Arch_NN) 
    
# s = y_test_mm.reshape(-1,1)     
# Arch_result.shape[1]        
    R2 = []   
    R2_t = []
    R2_scipy = []
    Poly_r2_score = []
    y_test_mm = np.genfromtxt(open("C:/Users/mark_/mark_data/Input/y_test_mm.csv"), delimiter=',')    
    Arch_1l = np.genfromtxt(open("C:/Users/mark_/mark_data/Input/Archopti_test_1_layer.csv"), delimiter=',')  #Predictions (Yhat_v_mlp_1_layer) made separately from test (X_test_mm) data in 1-layer model
    Arch_2_X = np.genfromtxt(open("C:/Users/mark_/mark_data/Input/Archopti_test.csv"), delimiter=',') 
    # Arch_2x_d = np.delete(Arch_2_X, [30, 40, 50, 60, 70, 71, 80, 81, 82, 89], axis =1)
    print('1-l',Arch_1l)
    print('2x',Arch_2_X) 
##    Arch_result = np.hstack((Arch_1l.T, Arch_2_X[0]))#Arch_2_X#Arch_2_X#  #Arch_2_X#Arch_2x_d, Arch_1l[:,None]
    Arch_result = np.c_[Arch_1l, Arch_2_X]#This works great!
    #np.savetxt("C:/Users/mark_/mark_data/Output/Arch_result.csv", Arch_result, delimiter=",", fmt="%10.4f")
    # Arch_1l.reshape(-1,1)

    plt.figure()
    plt.plot(Arch_result)
    plt.title('Arch result. should have number of peaks = number of layers (13)')
    plt.show()


    print('y_test_mm shape:',y_test_mm.shape) 
    print('Arch_result shape:',Arch_result.shape) 
    # y_test_mm_d = np.delete(y_test_mm, 92, axis =0)
    
    # Archopti_test_mse = np.genfromtxt(open("C:/Users/mark_/userdata/Output/Archopti_test_mse.csv"), delimiter=',')
    # Rank_idx = [sorted(Archopti_test_mse).index(x) for x in Archopti_test_mse]
    # from sklearn.linear_model import LinearRegression   
    # R2=[]
    for r in range(Arch_result.shape[1]):   # The number of columns = len(layer) x len(node). See h and n. 
##        reg_ao = LinearRegression(fit_intercept=True).fit(y_test_mm.reshape(-1, 1), Arch_result[:,r]) #.reshape(-1,1), fit_intercept=False forces intercept to zero, Yhat_polynomial
##        R2.append(reg_ao.score(y_test_mm.reshape(-1,1), Arch_result[:,r]))
        # reg_t = LinearRegression(fit_intercept=False).fit(y_test_mm.reshape(-1,1), Arch_result[:,r]) #.reshape(-1,1), fit_intercept=False forces intercept to zero, Yhat_polynomial
        # R2_t.append(reg_t.score(y_test_mm.reshape(-1,1), Arch_result[:,r]))
        R2_scipy.append(scipy.stats.linregress(y_test_mm, Arch_result[:,r]))
        #Poly_r2_score.append(sklearn.metrics.r2_score(y_test_mm[:,0], Arch_result[:,n]))#.reshape(-1,1),Yhat_poly_all.T[n].reshape(-1,1))) #, fit_intercept=True
        # plt.scatter(y_test_mm.reshape(-1,1), Arch_result[:,r], marker = ".")
        
##    R2_m_idx = np.argmax(R2)
##    R2_max = np.max(R2)
    rs = np.asarray(R2_scipy)#.reshape(-1,5)#, order='F'
    #rs = np.asarray(Poly_r2_score)#.reshape(-1,5)#, order='F'
    print('rs shape:', rs.shape)
    print('rs file. Order = slope, intercept, rvalue, pvalue, stderr, intercept_stderr:', rs)

    #print('rs 10,2', rs[10][2])  #This works to confirm the location of the rvalue in the R2_scipy array.
    r2_df = pd.DataFrame(data=R2_scipy)

    plt.figure()
    labels = ['r-value', 'slope']
    plt.plot(r2_df.loc[:,'rvalue'])  # This works
    plt.plot(r2_df.loc[:,'slope'])
    #plt.plot(np.asarray(R2_scipy).rvalue)
    #R2_scipy=np.asarray(R2_scipy).T
##    for i in range(len(rs)):  #len(np.asarray(R2_scipy).T)
##        plt.plot(rs[i][2])
##        i+=1
        #plt.plot(R2_scipy[i][2])
    plt.legend(labels)#labels)
##    plt.savefig('C:/Users/mark_/Documents/02_Professional/03_Unil/SATW/MDPI_paper/complementary_information/figures/Fig_mlp_layer_node_sl_rval.svg', format="svg")
    plt.show()

    max_rval = max(r2_df.loc[:,'rvalue'])
    print('Max rvalue', max_rval)
    print('Max R2 value', max_rval*max_rval)
    max_rindex = np.where(r2_df.loc[:,'rvalue']==max_rval)
    print('Max rval index', max_rindex)

    
    
##    rs_min = np.where(r2_df.loc[:,'rvalue']>0.6)# and r2_df.loc[:,'slope']>0.2)
##    print('rs passing value:',rs_min)
##    rs_pass = max(rs_min)
##    print('rs highest passing value:',rs_pass)
####    s_plus_r = [np.add(r2_df.loc[:,'slope'], r2_df.loc[:,'rvalue'])]
    
    s_plus_r = [np.add(r2_df.loc[:,'slope'], r2_df.loc[:,'rvalue'])]
    print('s_plus_r[0]',s_plus_r[0])
    max_val = max(s_plus_r[0])
    print('Max slope + rvalue', max_val)
    max_index = np.where(np.asarray(s_plus_r)==max_val)
    print('Max slope + rvalue index', max_index)
##    test = np.asarray(s_plus_r)
##    max_5 = np.argpartition(test, -5)[-5:]
##    #ind = np.argpartition(a, -4)[-4:]
##    #top4 = a[ind]
##    print('5 highest',test[max_5])

    plt.figure()
    plt.plot(np.asarray(s_plus_r[0]))  
    plt.title('slope + r-value')
    plt.show()
##
##    number_list = [1, 2, 3]
##
##max_value = max(number_list)
##
##Return the max value of the list
##
##
##max_index = number_list.index(max_value)
##
##Find the index of the max value
##
##
##print(max_index)
    #R2_mlp = scipy.stats.linregress(y_test_mm[:,0], Yhat_v_mlp[:,0])
##    print('R2 max:',R2_max) 
    print('number of columns:', Arch_result.shape[1])
##    print('R2 shape:',np.asarray(R2).shape)
    print('R2 scipy shape:',np.asarray(R2_scipy).shape)
    # m = R2
    ##R2 = np.square([r2_df.loc[:,'rvalue']])
    R2 = np.square([0.3 if (test < 0.3) else test for test in r2_df.loc[:,'rvalue']])
    print('R2 shape before imshow:', R2.shape)
    # np.savetxt("C:/Users/mark_/userdata/Output/Archopti_R2.csv",R2 , delimiter=",", fmt="%10.4f")
    # Opti=[]
    #print('R2 square(rvalue)', R2)
    #####  Image plot 9 x 8  (units x layers) ######
    no_nodes = 10#Arch_1l.shape[1]  # Read from satw_all.py. Number of layer elements + 1 = no_layers
    
    # z = np.array(R2).reshape(-1,no_nodes)
    #fig, ax = plt.subplots(figsize=(12,10))
    plt.figure(figsize=(12,12))  
    plt.imshow(np.array(R2).reshape(-1,no_nodes, order='C'))#  Verify that layer value is correct. (-1, len(Arch_result[1])), cmap='Spectral')  # Arch_1l.shapeax.imshow
    plt.yticks([0,1,2,3,4,5,6,7, 8, 9, 10, 11, 12], ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13"] ) # Hidden layers...in order to make neat ticks
    plt.ylabel("Number of layers")
    plt.xticks([0,1,2,3, 4, 5, 6, 7, 8, 9], ["8", "16", "32", "64", "128", "256", "512", "1024", "2048", "4096"] ) #Nodes (units) ...in order to make neat ticks
    plt.xlabel("Number of Nodes")
    plt.title("R${^2}$, validation test, 10 x 13 array (units x layers)")
    plt.colorbar()
##    plt.savefig('C:/Users/mark_/Documents/02_Professional/03_Unil/SATW/MDPI_paper/complementary_information/figures/Fig_mlp_layer_node_r2_10x13.svg', format="svg")
    plt.show()

    plt.figure(figsize=(12,12))  
    plt.imshow(np.asarray(s_plus_r).reshape(-1,no_nodes, order='C'))#  Verify that layer value is correct. (-1, len(Arch_result[1])), cmap='Spectral')  # Arch_1l.shapeax.imshow
    plt.yticks([0,1,2,3,4,5,6,7, 8, 9, 10, 11, 12], ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13"] ) # Hidden layers...in order to make neat ticks
    plt.ylabel("Number of layers")
    plt.xticks([0,1,2,3, 4, 5, 6, 7, 8, 9], ["8", "16", "32", "64", "128", "256", "512", "1024", "2048", "4096"] ) #Nodes (units) ...in order to make neat ticks
    plt.xlabel("Number of Nodes")
    plt.title("Sum of slope and r-value, validation test, 10 x 13 array (units x layers)")
    plt.colorbar()
##    plt.savefig('C:/Users/mark_/Documents/02_Professional/03_Unil/SATW/MDPI_paper/complementary_information/figures/Fig_mlp_layer_node_s_rval_10x13.svg', format="svg")
    plt.show() 
    
    return


    # #####   Drop out analysis   ######
    # # test_mse_rep = np.genfromtxt(open("C:/Users/mark_/userdata/Output/test_mse_rep.csv"), delimiter=',')
    # test_rms_rep = np.genfromtxt(open("C:/Users/mark_/userdata/Output/test_rms_rep.csv"), delimiter=',')
    
    # plt.figure()  #figsize=(12,10)
    # # plt.plot(test_mse_rep)
    # plt.plot(test_rms_rep[0:8], marker = "o", ms=10, linestyle = " ")
    # plt.ylabel("RMSE")
    # plt.xticks([0,1,2,3, 4, 5, 6, 7], ["0.1", "1", "2", "3", "4", "5", "10", "25"] ) #, "50"Nodes (units) ...in order to make neat ticks
    # plt.xlabel("Fraction of dropout units [%]")
    # plt.title("Effect of dropout (MLP, 4 layers, 256 units/layer)")
    # plt.savefig('C:/Users/mark_/Documents/02_Professional/03_Unil/SATW/MDPI_paper/MDPI_article/MDPI_article_v19/figures/rms_vs_do.svg', format="svg")
    # plt.show()
    
    # for r in range(len(Yhat_v_rep)):
    #         reg = LinearRegression(fit_intercept=False).fit(y_test_mm, Yhat_v_rep[r]) #.reshape(-1,1), fit_intercept=False forces intercept to zero, Yhat_polynomial
    #         R2.append(reg.score(y_test_mm, Yhat_v_rep[r]))
    #         # reg_t = LinearRegression(fit_intercept=False).fit(y_test_mm.reshape(-1,1), Yhat_v_rep[:,r]) #.reshape(-1,1), fit_intercept=False forces intercept to zero, Yhat_polynomial
    #         # R2_t.append(reg_t.score(y_test_mm.reshape(-1,1), Yhat_v_rep[:,r]))
    #         plt.scatter(y_test_mm, Yhat_v_rep[r], marker = ".")

###############  Tests for normality and skewness   ###########
def norm(y_data, y_data_z, y_data_t, y_data_1p, y_data_yj, y_data_mm, y_data_z_mm, y_data_t_mm, y_data_1p_mm, y_data_yj_mm):  #y_data_mm,y_data_t_mm,  y_data_1p_mm,, y_data_yj_mm,  y_data_z_mm
    # RD = pd.read_excel('C:\\Users\\mark_\\Anaconda3\Data\SATW_project.xlsx', sheet_name='Experiment_L9_V3') #sep=";",,  decimal=','Experiment_L4_V2
    # RD = pd.read_excel('C:\\Users\\mark_\\Anaconda3\Data\SATW_project.xlsx', sheet_name='Mock experiment_E9')
##    RD = pd.read_excel('C:\\Users\\mark_\\mark_data\Input\SATW_project.xlsx', sheet_name='Mock experiment_E9')
        # RD = pd.read_excel('/scratch/mmccormi1/SATW_project.xlsx', sheet_name='Experiment_L9_V3', index_col=None, engine="openpyxl") #sep=";",,  decimal=','Experiment_L4_V2
        # RD = np.genfromtxt(open("/scratch/mmccormi1/SATW_project.csv"), delimiter=';')
##    RD_Ex = pd.DataFrame(data = RD) # RD_Ex = reference data from the experiment with additional data derived from T and Q (170 data points) plus mock data created using equation X
        
##    CV_ref_all = np.multiply(RD_Ex.iloc[:170,[10]], RD_Ex.iloc[:170,[7]])  #  ref -> CV reduction derived from the "Experiment" (kJ/day). RD_Ex.iloc[7:226,38]
    # np.savetxt("C:/Users/mark_/userdata/Output/CV_ref_all.csv", CV_ref_all, delimiter=',')
    # y_data = pd.read_csv('C:/Users/mark_/userdata/Output/y_data.csv', sep=',', decimal='.', header=None, index_col=False)
    # y_data[y_data!=0]
    
    # data = y_data#np.asarray(y_data) #CV_ref_all
   
    #####  Normal distribution
    np.random.seed(42)
    norm_mm = np.random.normal(0.5, 0.125, 1530)
    cumsum_norm_mm = np.cumsum(norm_mm)
    # cumsum_norm_mm = (cumsum_norm[:] - cumsum_norm[:].min()) / (cumsum_norm[:].max() - cumsum_norm[:].min())
    #norm_mm = (norm[:] - norm[:].min()) / (norm[:].max() - norm[:].min())
    #cumsum_norm_mm = np.cumsum(norm_mm)
   
    ##### Mean, median and standard deviation    ######
    mean = np.mean(y_data)
    median = np.median(y_data)
    stdev = np.std(y_data)

    #####  raw data distribution
    cumsum_y = np.cumsum(y_data)
    # cumsum_y_mm = (cumsum_y[:] - cumsum_y[:].min()) / (cumsum_y[:].max() - cumsum_y[:].min())
    y_data_mm_r = (y_data[:] - y_data[:].min()) / (y_data[:].max() - y_data[:].min())
    cumsum_y_mm = np.cumsum(y_data_mm_r)
    
    
    #####  z-score transform distribution
    mean_z = np.mean(y_data_z)
    median_z = np.median(y_data_z)
    stdev_z = np.std(y_data_z)
    
    cumsum_z = np.cumsum(y_data_z)
    # cumsum_z_mm = (cumsum_z[:] - cumsum_z[:].min()) / (cumsum_z[:].max() - cumsum_z[:].min())
    #y_data_z_mm = (y_data_z[:] - y_data_z[:].min()) / (y_data_z[:].max() - y_data_z[:].min())
    cumsum_z_mm = np.cumsum(y_data_z_mm)
    
    ###  log transformed normalisation
    mean_t = np.mean(y_data_t)
    median_t = np.median(y_data_t)
    stdev_t = np.std(y_data_t)
    
    cumsum_t = np.cumsum(y_data_t)
    # cumsum_t_mm = (cumsum_t[:] - cumsum_t[:].min()) / (cumsum_t[:].max() - cumsum_t[:].min())
    #y_data_t_mm = (y_data_t[:] - y_data_t[:].min()) / (y_data_t[:].max() - y_data_t[:].min())
    cumsum_t_mm = np.cumsum(y_data_t_mm)
    
    #### log+1 transformed normalisation
    mean_1p = np.mean(y_data_1p)
    median_1p = np.median(y_data_1p)
    stdev_1p = np.std(y_data_1p)
    
    cumsum_1p = np.cumsum(y_data_1p)
    # cumsum_1p_mm = (cumsum_1p[:] - cumsum_1p[:].min()) / (cumsum_1p[:].max() - cumsum_1p[:].min())
    #y_data_1p_mm = (y_data_1p[:] - y_data_1p[:].min()) / (y_data_1p[:].max() - y_data_1p[:].min())
    cumsum_1p_mm = np.cumsum(y_data_1p_mm)
    
    ##  yj means Yeo-Johnson normailisation
    mean_yj = np.mean(y_data_yj)
    median_yj = np.median(y_data_yj)
    stdev_yj = np.std(y_data_yj)

    #### yj transformed normalisation
    #yd = y_data_yj + (-np.min(y_data_yj))
    cumsum_yj = np.cumsum(y_data_yj)  #yd
    # cumsum_yj_mm = (cumsum_yj[:] - cumsum_yj[:].min()) / (cumsum_yj[:].max() - cumsum_yj[:].min())
    #y_data_yj_mm = (y_data_yj[:] - y_data_yj[:].min()) / (y_data_yj[:].max() - y_data_yj[:].min())
    cumsum_yj_mm = np.cumsum(y_data_yj_mm)
    
    # print(np.min(data))
    # print(np.min(y_data))
    
    # transformed = np.concatenate([data, np.asarray(y_data_t), np.asarray(y_data_1p), y_data_bc], axis=1)
    
    ###  Cumulative Min-Max scaled response data (CVred) propbability plot (Kolmogorov–Smirnov tes)   
##    plt.figure(figsize=(12,10))
##    # label = ["Reference experiment",'Derived values']
##    # plt.plot(pd.DataFrame(data=HRT).iloc[:,[0]], linestyle="-") # Derived values,  linestyle=" ", marker='.', c="b"
##    plt.plot(cumsum_norm_mm, label="Normal")
##    plt.plot(cumsum_y_mm, label="Raw data")
##    plt.plot(cumsum_z_mm, label="z-score transformed")
##    plt.plot(cumsum_t_mm, label="log transformed")
##    plt.plot(cumsum_1p_mm, label="log(1+y) transformed")
##    plt.plot(cumsum_yj_mm, label = "Yeo-Johnson transformed")
##    plt.xlim([0,1600])
##    plt.ylim([0,1000])
##    plt.xlabel('Sample order (day)')
##    plt.ylabel('Predictions (CVred)')
##    plt.legend()
##    plt.title('Cumulative Min-Max scaled response data', fontsize='large')
##    #plt.savefig('C:/Users/mark_/Documents/02_Professional/03_Unil/SATW/MDPI_paper/complementary_information/figures/Fig_CumulCVred.svg', format="svg")
##    plt.show()
    
    Mean_res = pd.DataFrame([mean, mean_t, mean_1p, mean_yj, mean_z], index = ['mean', 'mean_t', 'mean_1p', 'mean_yj', 'mean_z'])#.transpose()
    Median_res = pd.DataFrame([median, median_t, median_1p, median_yj, median_z], index = ['median', 'median_t', 'median_1p', 'median_yj', 'median_z'])#.transpose()
    Stdev_res = pd.DataFrame([stdev, stdev_t, stdev_1p, stdev_yj, stdev_z], index = ['stdev', 'stdev_t', 'stdev_1p', 'stdev_yj', 'stdev_z'])#.transpose()
    ##  Shapiro-Wilk test for normality. HO = Sample was drawn from a normal distribution and p > alpha
    ##  p < alpha => not normal distributed.
    print('Shapiro-Wilk, skew, kurtosis, transformations stats')
    stat, p = shapiro(y_data)
    print('Raw sample, Shapiro-Wilk statistics=%.3f, p=%.8f' % (stat, p))
    # interpret
    alpha = 0.05
    if p > alpha:
    	print('Raw sample looks Gaussian (fail to reject H0)')
    else:
    	print('Raw sample does not look Gaussian (reject H0)')
    
    stat_t, p_t = shapiro(np.log(y_data))
    print('log transformed, Statistics=%.3f, p=%.8f' % (stat_t, p_t))
    # interpret
    alpha = 0.05
    if p_t > alpha:
    	print('Log transformed sample looks Gaussian (fail to reject H0)')
    else:
    	print('Log transformed sample does not look Gaussian (reject H0)')
    
    stat_1p, p_1p = shapiro(np.log1p(y_data))
    print('Log(1+y) transformed, Statistics=%.3f, p=%.8f' % (stat_1p, p_1p))
    # interpret
    alpha = 0.05
    if p_1p > alpha:
    	print('Log +1 transformed sample looks Gaussian (fail to reject H0)')
    else:
    	print('Log +1 transformed sample does not look Gaussian (reject H0)')
          
    stat_yj, p_yj = shapiro(y_data_yj)
    print('Y-J transformed, Statistics=%.3f, p=%.8f' % (stat_yj, p_yj))
    # interpret
    alpha = 0.05
    if p_yj > alpha:
    	print('Yeo-Johnson transformed sample looks Gaussian (fail to reject H0)')
    else:
    	print('Yeo-Johnson transformed sample does not look Gaussian (reject H0)')
        
    stat_z, p_z = shapiro(y_data_z)
    print('z-score transformed, Statistics=%.3f, p=%.8f' % (stat_z, p_z))
    # interpret
    alpha = 0.05
    if p_z > alpha:
        print('z-score transformed sample looks Gaussian (fail to reject H0)')
    else:
     	print('z-score transformed sample does not look Gaussian (reject H0)')
           
    SW_res = pd.DataFrame([stat, stat_t, stat_1p, stat_yj, stat_z], index=['SW','SW_t', 'SW_1p', 'SW_yj', 'SW_z'])#.transpose()
    
    ####### Kurtosis   ############
    # Pearson definition used ==> Normal = 3.0
    # Fisher definition used ==> Normal = 0.0.  negative => fatter than normal, asymptote goes to zero faster than normal. Positive => thinner than normal, asymptote goes to zero slower than normal.
    kur = kurtosis(y_data, fisher=True)
    print('Kurtosis, raw data, Fisher definition => Normal = 0.', kur)
    kur_t = kurtosis(np.log(y_data), fisher=True)
    kur_1p = kurtosis(np.log1p(y_data), fisher=True)
    kur_yj = kurtosis(y_data_yj, fisher=True)
    kur_z = kurtosis(y_data_z, fisher=True)
    # K_res = np.zeros(shape=(1530))
    K_res = pd.DataFrame([kur, kur_t, kur_1p, kur_yj, kur_z ], index=['kur','kur_t', 'kur_1p', 'kur_yj', 'kur_z' ])#.transpose()  kur_yj[0]
    # K_res.columns(['kur','kur_t', 'kur_1p', 'kur_bc' ])
    ####### Skewness   ############
    # Normal distribution => skewness = 0. > 0 => right tail. < 0 => left tail
    skew = scipy.stats.skew(y_data)
    print('Skew, raw data', skew)
    skew_t = scipy.stats.skew(np.log(y_data))
    skew_1p = scipy.stats.skew(np.log1p(y_data))
    skew_yj = scipy.stats.skew(y_data_yj)
    skew_z = scipy.stats.skew(y_data_z)
    S_res = pd.DataFrame([skew, skew_t, skew_1p, skew_yj, skew_z ], index=['skew','skew_t', 'skew_1p', 'skew_yj', 'skew_z' ])#.transpose()
    
    # Sum_res = pd.concat([Mean_res.iloc[:,0], Median_res.iloc[:,0], Stdev_res.iloc[:,0], SW_res.iloc[:,0], K_res.iloc[:,0], S_res.iloc[:,0]], axis = 1, ignore_index=True, join='outer')#
    Sum_res = np.concatenate([Mean_res.iloc[:,0], Median_res.iloc[:,0], Stdev_res.iloc[:,0], SW_res.iloc[:,0], K_res.iloc[:,0], S_res.iloc[:,0]]).reshape((5,-1), order='F')#
    Sum_res_df = pd.DataFrame(Sum_res, index=['data', 'log', 'log(1+y)','Yeo-Johnson', 'z-score'], columns= ['Mean', 'Median', 'Stdev','S-W stat','Kurtosis','Skew'])
    #Sum_res_df.to_csv("C:/Users/mark_/mark_data/Output/V24/Summary_result_resp.csv", sep=",", float_format="%10.4f", header=['Mean', 'Median', 'Stdev','S-W stat','Kurtosis','Skew'])
    #np.savetxt("C:/Users/mark_/mark_data/Output/V24/Summary_result_resp.csv", Sum_res_df, delimiter=",", fmt="%10.4f") # non-shuffled data
    # print(pd.DataFrame(transformed).describe())
    # L9_desc_stat=l9_CVred.describe().transpose()
    # # L9_skew = pd.DataFrame(scipy.stats.skew(l9_CVred, axis=0, bias=True, nan_policy='propagate'))
    # # L9_kurtosis = pd.DataFrame(scipy.stats.kurtosis(l9_CVred, axis=0, fisher=True, bias=True, nan_policy='propagate'))
    # L9_skew = l9_CVred.skew(axis=0)
    # L9_kurtosis = l9_CVred.kurtosis(axis=0)
    # L9_median = l9_CVred.median(axis=0)
    # frame_stats= pd.concat([L9_kurtosis, L9_skew, L9_median]  , axis=1, ignore_index=False)  
    # # L9_stats_CVred = frame_stats.append(L9_desc_stat,ignore_index=True, verify_integrity=True, sort=False)#, sort=True) #, L9_skew, ignore_index=True
    # L9_stats_CVred = pd.concat([L9_desc_stat, frame_stats] , axis=1, ignore_index=False, verify_integrity=True, sort=True, copy=False) #, ignore_index=TrueL9_stats_CVred.merge(right=L9_skew, how='left') #, L9_skew
    # L9_stats_CVred.columns = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max', 'kurtosis', 'skew', 'median'] 

    return 

# #######  AFBR characteristics  ############
    # AFBR total volume (liquid below, filter, liquid above) [l]: 56 
    # AFBR liquid volume (liquid above + below + packing void space)[l]: 37
    # #Vessel inner diameter [mm]
    # #Vessel cross section [m2]: 0.363
    # #Packed bed Reynolds number
    # #Density of the fluid [kg/m3]: 1050
    # #Superficial velocity [m/s]: V_l
    # #Spherical diameter of particles [m]: S:0.006 M:0.012 L:0.036
    # #Dynamic viscosity, μ [N*s/m2]: 0.003
    # #Average volume of 1 void space [ml] S:0.013 M:1.10 L:28
    # #Average radius of 1 void space [m] S:0.00145   M: 0,00645 L:0.0188
    
    # print('COMMENTED PRESENTATION OF EXPERIMENTALLY OBTAINED RESULTS COMPLIMENTED WITH SURROGATE DATA')

#######    MDPI statistics    ###################33
def satw_ks_stats(y_train_mm, y_test_mm):
  ######## K-S statistic    ###################### The Kolmogorov–Smirnov statistic quantifies a distance between the empirical distribution functions of 
  ##two samples. The null hypothesis is that the samples are drawn from the same distribution (in the two-sample case).
  ## See Wikipedia
    
    y_train_mm = np.array(y_train_mm)
    y_test_mm = np.array(y_test_mm)
    m = len(y_train_mm)
    n = len(y_test_mm)
    
    # y_train_mm_k, y_test_mm_k = train_test_split(y_train_mm, test_size=0.3, random_state= 42, shuffle = True, stratify= None)
    # y_train_mm = np.array(y_train_mm_k)
    # y_test_mm = np.array(y_test_mm_k)    
    # m = len(y_train_mm_k)
    # n = len(y_test_mm_k)
    
    alpha = 0.001  #  1- level of confidence/100
    D_ks = sqrt(-np.log(alpha/2)*((1+m/n)/(2*m)))  # The Kolmogorov-Smirnov test critical value. The null hypothesis is rejected at level alpha if the calculated K-S statistic is > D_ks. 
    
    outputs_k = []  # 1071 train + 459 test samples = 1530 samples
    for sublist in [[i] for i in y_train_mm.flatten()]:  #Yhat_test_mm_4e
        for item in sublist:
            outputs_k.append(item)
    outputs_test_k = []
    for sublist in [[i] for i in y_test_mm.flatten()]: 
        for item in sublist:
            outputs_test_k.append(item)
    statistic, pvalue =ks_2samp(outputs_k, outputs_test_k, mode='exact')  #returns ks statistic and p-value. mode = exact or auto
    print('K-S alpha value',alpha)
    print("D_ks value, Critical value", D_ks)  #Critical value
    print("K-S statistic", statistic)
    print("K-S p-value", pvalue)
    if statistic > D_ks:
        print("Reject the null hyposthesis. The training and validation datasets have different distributions. alpha level =", alpha)
    else:
        print("The training and validation datasets have the same distribution. alpha Level =", alpha)
    
    return

###############  Autocorrelation  analysis ####################33
def autocorr(CV_ref_kjd, X_data):
    # import numpy as np
    # import pandas as pd
    # import matplotlib.pyplot as plt
    # # from pandas import read_csv
    # from matplotlib import pyplot
    # #import statsmodels.graphics.tsaplots # as splt
    # from statsmodels.graphics.tsaplots import plot_pacf, plot_acf 
    # #from statsmodels.graphics.tsaplots import plot_acf
    import statsmodels
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from statsmodels.graphics.tsaplots import plot_pacf, plot_acf   
    from statsmodels.stats.stattools import durbin_watson#, acf
   

    #from statsmodels.graphics.tsaplots import plot_acf
    
    #series = read_csv("C:/Users/mark_/tempuser/daily-min-temperatures.csv", header=0, index_col=0)  #Test\
    # dates = pd.date_range(start = '4-5-2017', end = '12-12-2017', freq = 'D') 
    # series = CV_ref_all   
    #series = read_csv("C:/Users/mark_/tempuser/daily-min-temperatures.csv", header=0, index_col=0) 
    # Adding plot title.
    # plt.title("Autocorrelation Plot")
    # # Providing x-axis name.
    # plt.xlabel("Lags")
     
    ##### Plotting the Autocorrelation plot.
    data=CV_ref_kjd.flatten()  #to_numpy().
    # plt.acorr(data, maxlags = 100, usevlines=False, normed=False)
    
    
    dw = durbin_watson(data, axis=0)
    print('Durbin_Watson test statistic', dw)
    # https://www.statsmodels.org/dev/generated/statsmodels.stats.stattools.durbin_watson.html
    # The test statistic is approximately equal to 2*(1-r) where r is the sample autocorrelation of the residuals. 
    # Thus, for r == 0, indicating no serial correlation, the test statistic equals 2. 
    # This statistic will always be between 0 and 4. 
    # The closer to 0 the statistic, the more evidence for positive serial correlation. 
    # The closer to 4, the more evidence for negative serial correlation.
    
    from statsmodels.tsa.stattools import acf
    acf, confint, qstat, pvalues = acf(data, adjusted=False, nlags=100, qstat=True, fft=True, alpha=0.05)#, bartlett_confint=True, missing='none')
    print(np.max(pvalues))
    
    plt.figure(figsize=(10,15))
    plot_acf(CV_ref_kjd, lags=100, alpha=0.05)  #alpha =1-CI
    plt.xlabel('Lag time(days)')
    plt.ylabel('Correlation coefficient(R2)')
    #plt.legend()
    plt.title('Autocorrelation of lagged calorific value reductions', fontsize='large')
##    plt.savefig('C:/Users/mark_/Documents/02_Professional/03_Unil/SATW/MDPI_paper/complementary_information/figures/Fig_autocorrelation.svg', format="svg")  #MDPI_article/MDPI_article_v24
    plt.show() #pyplot
    
    #return
    
    ###### Variance inflation factor and Pearson correlation between predictors  ######################
    # Test for multicolinearity in the predictor dataset. 
    # See satw_data.py line 1838 for a reference to development of the 4th degree polynomial.
    # Predictor dataset to test for colinearity: X_train_mm
    # Predictor dataset X_train_mm is saved on line 1696 of satw_data.py
##    from statsmodels.compat.python import lzip
##    from statsmodels.compat.pandas import Appender
##    from statsmodels.graphics._regressionplots_doc import _plot_influence_doc
##    from statsmodels.regression.linear_model import OLS
##    from statsmodels.stats.multitest import multipletests
##    from statsmodels.tools.decorators import cache_readonly
##    from statsmodels.tools.tools import maybe_unwrap_results
##    # import statsmodels.api as sm
##    
##    
##    statsmodels.tools.add_constant
    # pr = pd.read_csv("C:/Users/mark_/userdata/Output/X_data.csv", header=None, index_col=None) #X_data_mm"Mat, HDR, SPH, Qin"Predictors
    # pr.insert(0, "const", 1)
    # pr = []
    pr = X_data
    print(X_data.shape)
    
    # pr = pr.to_numpy(dtype='float32')#X_data_mm#
    # Predictor_idx = 0.0#Predictors[1.0]
    # exog_idx = 1.0
    vif = [statsmodels.stats.outliers_influence.variance_inflation_factor(pr, i) for i in range(pr.shape[1])]  #, exog_idx
    print("Variance Inflation Factors",vif)
    ###  If VIF > 4 the the variable is highly collinear. Range = 0 to >10 

   
    
    # model = sm.OLS(y, X)
    # results = model.fit()
    # print(results.summary())
    # dw = statsmodels.stats.stattools.durbin_watson(resids, axis=0)
    
    #float_formatter = "{:.2f}".format
    pearson = []
    R=[]
    P = []
    i=0
    j=0
    while j < 4:
        #print(i,j)
        for i in range(4):
            print(i,j)
            x = pr[:,i]
            y = pr[:,j]
            rr, pp = scipy.stats.pearsonr(x, y)
            pearson.append(scipy.stats.pearsonr(x,y))
            R.append(rr)
            P.append(pp)
        j+=1
        i=0
    pear = np.reshape(pearson,(-1,4), order='c')
    print("r", pear)
    print('Autocorr pearson r',R)
    print('Autocorr pearson p',P)
##            for j in range(5):
    # #             y = pr[:,2]
    return
    # [scipy.stats.pearsonr(x,y) for i in range(pr.shape[0])]
    # x = prNS_res["y_test_mm_ns"].corr(NS_res["Yhat_polynomial_m_ns"], method = 'pearson')
    # y = pr[:,3]
    # while i < 5:
    #     Px = [scipy.stats.pearsonr(pr[:,i],pr[:,(i+1)]) for i in range(4)]
    #     i+=1
    # d=0
    # days = [d+1 for d in range(170)]    
    # x = pr[:,1]
    # y = pr[:,3]
    # xdf = pd.DataFrame(x)
    # ydf = pd.DataFrame(y)
    # r, pvalue = scipy.stats.pearsonr(x,y)
    # plt.plot(x,y)#, linestyle=" ", marker='.', c="b")
    # plt.plot(y)
    # plt.show()
    # plt.plot(xdf.values,ydf.values)
    # scipy.stats.pearsonr(x,y)
##    prDF = pd.DataFrame(pr)
##    
##    prDF[0].corr(prDF[3], method = 'pearson')
##    # pd.DataFrame.plot(pr.iloc[0],pr.iloc[3])
##    pr.plot()
##    pr.plot.line(x=1, y=3)

                   # from scipy import stats

# a = np.array([0, 0, 0, 1, 1, 1, 1])
# a = np.array([0, 2, 2.5, 3.1, 4.1, 5.2, 6])
# b = np.arange(7)
# plt.scatter(a,b)
# rr, pp = scipy.stats.pearsonr(a, b)

# np.round((alpha),decimals=1)

# Yhat_test_mm_4e = np.genfromtxt('C:/Users/mark_/userdata/Output/Yhat_v_mlp_4e.csv', delimiter=',')
# head = Yhat_test_mm_4e[:5]
# Mean = np.mean(Yhat_test_mm_4e, axis = 0)
# Std = np.std(Yhat_test_all, axis = 0)

#######    MDPI Figures    ###################

## Import data for figures
# Import saved outputs from the MLP models. Attention figure 4 is now figure 2.....
# Yhat_v = pd.read_csv('C:/Users/mark_/userdata/Output/Yhat_v.csv', sep=',', decimal='.', header=None, index_col=False)#, axis=1) 
# y_test_mm = pd.read_csv('C:/Users/mark_/userdata/Output/y_test_mm.csv', sep=',', decimal='.', header=None, index_col=False)#, axis=1) 
# y_test_mm_t = pd.read_csv('C:/Users/mark_/userdata/Output/y_test_mm_t.csv', sep=',', decimal='.', header=None, index_col=False)#, axis=1)  # True unshuffled data 
# y_test_mm_lstm = pd.read_csv('C:/Users/mark_/userdata/Output/y_test_lstm_mm.csv', sep=',', decimal='.', header=None, index_col=False)#, axis=1)  # True unshuffled data. tflearn results = y_test_mm_lstm.csv 
# yhat_test_mm_lstm = pd.read_csv('C:/Users/mark_/userdata/Output/Yhat_test_lstm_mm.csv', sep=',', decimal='.', header=None, index_col=False)
# Yhat_test_mm_4b = np.mean(pd.read_csv('C:/Users/mark_/userdata/Output/Yhat_v_mlp_4d_do5.csv', sep=',', decimal='.', header=None), axis=1)
    
# # #######  Figure 5 MDPI V21    ######
# # # To make this figure do preprocessing separately.

# plt.figure(figsize=(7,6))   
# # legend = ["Polynomial"]
# plt.plot(xl,yl,  c="k", linestyle='solid')
# # plt.scatter(y_test_mm, Yhat_polynomial_m, linestyle='-', c="g", s=2, label='Polynomial')
# plt.scatter(outputs_test,Yhat_v, linestyle='-', c="g", s=2, label='LSTM')
# # axc.set_title('4c 6-layer perceptron, shuffled', fontsize=10)
# plt.legend()#['True outputs (Test data)', 'Predictions (Test data)', 'Predictions (Test-2 data)'])
# plt.xlabel('True values') #Measurement sequence number
# plt.ylabel('Predictions')
# plt.title('MLP model - preshuffled data, predicted and true values, LSTM validation test  Daily CV reduction (MinMax scaled)', fontsize='large') #MLP predicted and observed flow rate during testing
# plt.show()
# # plt.savefig('C:/Users/mark_/Documents/02_Professional/03_Unil/SATW/MDPI_paper/MDPI_article/MDPI_article_v21/figures/Fig_TvP.svg', format="svg")

### # # # Plot a slope of 1 to show a perfect correlation between predictions and true values
##yl = np.array([0,1,0.1])
##xl = np.array([0,1, 0.1])  
##
##fig, ax = plt.subplots(figsize=(12,10))
##plt.figure(figsize=(12,10))
### label = ["Reference experiment",'Derived values']
##plt.plot(xl,yl,  c="k", linestyle='solid')
### plt.scatter(y_test_mm, Yhat_polynomial_m, linestyle='-', c="g", s=2, label='Polynomial')
### plt.scatter(y_test_mm_lstm,yhat_test_mm_lstmv, linestyle='-', c="g", s=2, label='LSTM')
### plt.plot(Merge_exp.iloc[:,[1]], linestyle=" ", marker='D', c="r")  # Reference experience
### plt.plot(Merge_exp.iloc[:,[0]], linestyle=" ", marker='.', c="b") # Derived values
### plt.plot(pd.DataFrame(data=HRT).iloc[:,[0]], linestyle="-") # Derived values,  linestyle=" ", marker='.', c="b"
##plt.xlabel('True values')
##plt.ylabel('Predictions')
##plt.legend()
##plt.title('Calorific value reduction, Predicted verses True values', fontsize='large')
### plt.savefig('C:/Users/mark_/Documents/02_Professional/03_Unil/SATW/MDPI_paper/MDPI_article/MDPI_article_v21/figures/Fig.svg', format="svg")
##plt.show()





##def satw_figures(y_test_mm):#), y_test_mm_4c):#),, y_test_mm_4b y_test_mm_4b, Yhat_polynomial): #CV_ref_all, IFR_ref_all, 
#####    Import surrogate data    ###################33
##
##    # Surr_data = pd.read_excel('C:\\Users\\mark_\\Anaconda3\Data\SATW_project.xlsx', sheet_name='Experiment_L4_V2') #sep=";",,  decimal=',''Experiment_L4_V2'
##    Exp_data_CVred = pd.read_excel('C:\\Users\\mark_\\Anaconda3\Data\SATW_project.xlsx', sheet_name='CVred') 
##    Exp_data_inflow = pd.read_excel('C:\\Users\\mark_\\Anaconda3\Data\SATW_project.xlsx', sheet_name='import') 
##    # dates = (["May", "June", "July", "August", "September", "October", "November", "December"])
    
# dFmt = mdates.DateFormatter('%d')

########   Visualize raw data  ###########
# from sklearn.decomposition import PCA
# from sklearn.cross_decomposition import PLSRegression, PLSCanonical
# from sklearn.linear_model import LinearRegression
# #PCA
# TSD = np.concatenate(([4]*170,[4]*170,[36]*170,[36]*170))
# TMAT = np.concatenate(([1]*170,[10]*170,[1]*170,[10]*170))
# THD = np.concatenate(([0.5]*170,[4]*170,[4]*170,[0.5]*170))

# plt.plot(TSD)

# X = np.array([TSD,TMAT,THD], dtype=object) #, []
# X= np.transpose(X)
#y= np.concatenate((Surr_data.loc[:169, ["E1 CH4 [kJ/day]"]], Surr_data.loc[:169,["E2 CH4 [kJ/day]"]],
#                        Surr_data.loc[:169, ["E3 CH4 [kJ/day]"]], Surr_data.loc[:169, ["E4 CH4 [kJ/day]"]]))
#plt.hist(y, bins = 25)
#plt.plot(y)
# pca = PCA(n_components=3, whiten=True)# Whiten = False is the dfault value. True "ensure uncorrelated outputs with unit component-wise variances"
# pca.fit(X,y)

# plt.scatter(X[:,0],X[:,1], alpha=0.9, label = "samples")
# pca.fit_transform(X,y)
# pca.inverse_transform(X,y)
# print(pca.components_)
# print(pca.explained_variance_)
# print(pca.explained_variance_ratio_)
# print(pca.singular_values_)
# print(pca.mean_)
# print(pca.n_components_)
# print(pca.n_features_)
# print(pca.n_samples_)
# plt.scatter(pca.n_features_, pca.mean_)
# plt.plot(pca.components_)
# plt.plot(pca.fit_transform(X,y))
# print (TSD)
# plt.plot(y)


# plsca = PLSCanonical(n_components=3)
# plsca.fit(np.transpose(X),y)
# print(plsca.x_rotations_)
# plsca.inverse_transform(X)

# pls = PLSRegression(n_components=1)
# pls.fit(np.transpose(X), y)
# plt.plot(pls.transform(X))

# print(pls.x_weights_)
# print(pls.x_loadings_)
# print(pls.x_scores_)
# print(pls.y_loadings_)
# plt.scatter(pca.n_features_, pca.mean_)


# reg = LinearRegression().fit(X, y)
# print(reg.score(X, y))
# print(reg.coef_)
# print(reg.intercept_)


########   Import experimental CV reduction (kJ/l) and derived CV reduction  ###########
##    DCVr = Exp_data_CVred.iloc[:,:2].set_index('Date_', inplace=False) # derived experimental CV reduction (kJ/l) using equation 5. Exp_data_CVred.iloc[:,:2].set_index('Date_', inplace=False)
##    DCVo = Exp_data_CVred.iloc[:,3:5].set_index('Date_Exp', inplace=False) # original experimental CV reduction (kJ/l) using equation 5
##    Date_ = Exp_data_CVred.iloc[:,5]
##    Merge_exp = DCVr.join(DCVo)
    
########   Import experimental influent flow rate (l/day)  ###########
    Qr = Exp_data_inflow.iloc[:,[3]]#.set_index('Date_', inplace=False) # 
    plt.plot(Qr)
    # DCVo = Exp_data_inflow.iloc[:,3:5].set_index('Date_Exp', inplace=False) # original experimental CV reduction (kJ/l) using equation 5
    # Date_ = Exp_data_inflow.iloc[:,0]
    # Merge_exp = DCVr.join(DCVo)    

####  Dates for figure 2   #########
    # base = datetime.datetime(2017, 5, 4)
    # dates = [base + datetime.timedelta(days=i) for i in range(226)]  # May 4 to December 15th = 226 days

########   Import experimental and derived CV reduction (kJ/l)   ###########
# CV_der = Surr_data.iloc[:170,[10]] # derived CV reduction (kJ/l) using equation 5. Reference experiment
# CV_der =CV_der.replace(0, np.NaN)

########    Hydraulic Retention time (hours) and calorific value reduction (kJ/day)    #############
    # CV_red = np.multiply(Surr_data.iloc[:170,[10]], Surr_data.iloc[:170,[7]]) # Reference CV reduction (kJ/l), experiment * IFR = kJ/day        
    # CV_red=CV_red.replace(0, np.NaN)
    # plt.plot(CV_red)
    # CV_red = Merge_exp

    # HRT = np.divide(37, (np.divide(Surr_data.iloc[:,[7]], 24)))
    HRT = np.divide(37, (np.divide(Qr, 24)))
    # HRT=HRT.replace(0, np.NaN)  # AFBR total volume (liquid + packing): 56 liters, AFBR liquid volume (liquid above + below + packing void space): 37 liters
    # plt.plot(HRT)# VVD = Surr_data.iloc[:,[1]]/56

# D = pd.DataFrame(Surr_data.iloc[:,[37]])
# D['M-D'] = D.Date.dt.month.astype(str).str.cat(D.Date.dt.day.astype(str), sep="-")
# D['CH4'] = Surr_data.iloc[:,[89]]
# # D['CH4_vvd'] = Surr_data.iloc[:,[89]]/56

# np.savetxt("C:/Users/mark_/Documents/02_Professional/03_Unil/SATW/MDPI_paper/MDPI_article/Data/CH4_from_PCI", D.values, fmt='%d')
# CH4_from_PCI = D.to_csv(path_or_buf="C:/Users/mark_/Documents/02_Professional/03_Unil/SATW/MDPI_paper/MDPI_article/Data/CH4_from_PCI.csv", sep=";", decimal=",", columns=['Date','M-D', 'CH4','CH4_vvd']) #, sep=","
# D_exp = pd.DataFrame(Exp_data.iloc[12:22,[0]]) #12:22
# D_exp['Date'] = pd.to_datetime(D_exp['Date'])
# D_exp['M-D'] = D_exp.Date.dt.month.astype(str).str.cat(D_exp.Date.dt.day.astype(str), sep="-") #['M-D'], errors='coerce'
# MDPI_fig_1= np.concatenate([D, HRT, VVD], axis = 1)
# np.savetxt('C:\\Users\\mark_\\Anaconda3\Data\MDPI_fig_1.csv', MDPI_fig_1, delimiter=",")
# np.savetxt('C:\\Users\\mark_\\Anaconda3\Data\MDPI_fig_1.txt', MDPI_fig_1, delimiter=",")
# np.savetxt('C:\\Users\\mark_\\Anaconda3\Data\MDPI_fig_1_date.csv', D)#, delimiter=","

# plt.figure(figsize=(16,6))
# # plt.plot(D['M-D'], HRT, linestyle='-', c='r')#, label = "Hydraulic Retention Time [days]")
# # plt.plot(D['M-D'], VVD, linestyle='-', c='b')#, label = "Methane production [vessel volumes.day$^-$$^1$]")
# # plt.plot(D['M-D'], CH4_PCI, marker='o', c='g', markersize=12) #['M-D']

# ax1=HRT.replace(0, np.NaN).plot( linestyle='-', c='r', label = "Hydraulic Retention Time [hour]")
# ax2 = ax1.twinx()
# CV_red.replace(0, np.NaN).plot(ax = ax2, linestyle='-', c='b', secondary_y=True, label = "Calorific value reduction [MJ.day$^-$$^1$]") #

# plt.xticks(np.arange(0,230, step=30.5), ('May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'), rotation=30)#, 
# plt.legend()
# # plt.savefig('C:/Users/mark_/Documents/02_Professional/03_Unil/SATW/MDPI_paper/MDPI_article/MDPI_article_v03/figures/F2_hrt_ch4.svg', format="svg")
# plt.show() 


################## MDPI Figure 1  ##############
#Experimentally acquired data and derived data (kJ/l)
    fig, ax = plt.subplots(figsize=(12,10))
    plt.figure(figsize=(12,10))
    label = ["Reference experiment",'Derived values']
    plt.plot(Merge_exp.iloc[:,[1]], linestyle=" ", marker='D', c="r")  # Reference experience
    plt.plot(Merge_exp.iloc[:,[0]], linestyle=" ", marker='.', c="b") # Derived values
    plt.plot(pd.DataFrame(data=HRT).iloc[:,[0]], linestyle="-") # Derived values,  linestyle=" ", marker='.', c="b"
    plt.xlabel("Date")
    plt.ylabel("Calorific value reduction (kJ.liter-1)")
    plt.ylim([0.05,0.5])
    plt.legend(label)
    plt.title('Experimental results', fontsize='large')
# # plt.savefig('C:/Users/mark_/Documents/02_Professional/03_Unil/SATW/MDPI_paper/MDPI_article/MDPI_article_v11/figures/F1_experiment.svg', format="svg")
    plt.show()


################## MDPI Figure 2  ##############
##F1_hrt_cv reduction

    # fig, ax = plt.subplots(figsize=(12,10))
    # # lims = [(np.datetime64('2017-05-04'), np.datetime64('2017-12-15'))]
    # # label =(["Hydraulic Retention Time [hour]","Calorific value reduction [MJ.day$^-$$^1$]"]["Hydraulic Retention Time [hour]","Calorific value reduction [MJ.day$^-$$^1$]"])
    # # label2 =(["Hydraulic Retention Time [hour]","Calorific value reduction [MJ.day$^-$$^1$]"]["Hydraulic Retention Time [hour]","Calorific value reduction [MJ.day$^-$$^1$]"])
    # # label = ["Reference experiment",'Derived values']
    # # H=ax.plot(Date_, HRT, linestyle='-', c='k') #dates, label = "Hydraulic Retention Time [hour]"
    # ax.set_ylabel("Hydraulic Retention Time [hours]")#, color="k")#, fontsize=10)
    # ax.set_xlabel("Date")#, color="k")#, fontsize=10)
    # # ax.legend()
    # ax2 = ax.twinx()
    # # C=ax2.plot(Date_, CV_red, linestyle='-', c='k', label = "Calorific value reduction [kJ.day$^-$$^1$]")
    # C=ax2.plot(Date_,Merge_exp.iloc[:,[1]], linestyle=' ',  marker='D', c='r', label = "Reference experiment")#.re
    # D=ax2.plot(Date_,Merge_exp.iloc[:,[0]], linestyle=' ', marker='.', c='b', label = "Derived values")#.replace(0, np.NaN)dates 
    # ax2.set_ylabel("Calorific value reduction [kJ.day$^-$$^1$]",color="k",fontsize=10)
    # lns=C+D
    # labs=[l.get_label() for l in lns]
    # ax.legend(lns,labs) #
    
    # # lns=H+C
    # # labs=[l.get_label() for l in lns]
    # # ax.legend(lns,labs) #
    # # H1=[l.get_label() for l in H]
    # # C1=[l.get_label() for l in C]
    # # ax.legend(H1, C1)
    # # ax2.legend()
    # # plt.legend([" Hydraulic Retention Time [hour]", "Calorific value reduction [MJ.day$^-$$^1$]"])
    # # plt.xaxis.set_major_formatter(dFmt)
    # ax.set_xticklabels(Date_.datetime.replace(year=0, minute=0, second=0)) # Date, , rotation = 30
    # ax.xticks(np.arange(0,170, step=21), ('May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'), rotation=30)

    # #plt.savefig('C:/Users/mark_/Documents/02_Professional/03_Unil/SATW/MDPI_paper/MDPI_article/MDPI_article_v12/figures/F2_hrt_cv_red_v2.svg', format="svg")
    # plt.show()
    
    # #### Correlation of CV reduction with HRT   ######
    # print(len(HRT))
    # print(len(CV_red))
    
    # fig, ax = plt.subplots(figsize=(12,10))
    # ax.scatter(HRT.iloc[:,0], CV_red.iloc[:,0]) 
    # ax.set_xlabel("Hydraulic Retention Time [hours]")#, color="k")#, fontsize=10)
    # ax.set_ylabel("CV reduction")#, color="k")#, fontsize=10)
    # plt.show()
    
    # reg = sklearn.linear_model.LinearRegression(fit_intercept=True).fit(HRT.iloc[:,0].to_numpy().reshape(-1, 1), CV_red.iloc[:,0].to_numpy().reshape(-1, 1)) # Set fit_intercept to False to force 0. , normalize=False
    # print("4d, Validation test R2: %.2f" % reg.score(HRT.iloc[:,0].to_numpy().reshape(-1, 1), CV_red.iloc[:,0].to_numpy().reshape(-1, 1)))
    
    # fig, ax = plt.subplots(figsize=(12,10))
    # ax.histogram(CV_red.iloc[:,0], HRT.iloc[:,0]) 
    # ax.set_ylabel("Hydraulic Retention Time [hours]")#, color="k")#, fontsize=10)
    # ax.set_xlabel("CV reduction")#, color="k")#, fontsize=10)
    # plt.show()
    
################## MDPI Figure 4a, 4b, 4c and 4d   #####################
# #     # fig 4a = polynomial with pre-shuffled data (for code see next section and file satw_mlp_v5)
# #     # fig 4b = 1-layer MLP with pre-shuffled data (set shuffle to True in NN_preprocess function, satw_data file)
# #     # fig 4c = 2-layer MLP with pre-shuffled data (set shuffle to True in NN_preprocess function, satw_data file)
# #     # fig 4d = 3-layer MLP with pre-shuffled data (set shuffle to True in NN_preprocess function, satw_data file)
# #     # fig 4e = 6-layer MLP with pre-shuffled data (set shuffle to True in NN_preprocess function, satw_data file)
# #     # fig 4f = 6-layer MLP with UNSHUFFLED data (set shuffle to False in NN_preprocess function, satw_data file)

   # Yhat_v_mlp_4_e_t
# # # Plot a slope of 1 to show a perfect correlation between predictions and true values
    yl = np.array([0,1]) #,0.1
    xl = np.array([0,1]) #,0.1

# Import saved outputs from the MLP models. Attention figure 4 is now figure 2.....
    Yhat_v = pd.read_csv('C:/Users/mark_/userdata/Output/Yhat_v.csv', sep=',', decimal='.', header=None, index_col=False)#, axis=1) 
    # y_test_mm = pd.read_csv('C:/Users/mark_/userdata/Output/y_test_mm.csv', sep=',', decimal='.', header=None, index_col=False)#, axis=1) 
    y_test_mm_t = pd.read_csv('C:/Users/mark_/userdata/Output/y_test_mm_t.csv', sep=',', decimal='.', header=None, index_col=False)#, axis=1)  # True unshuffled data 
    y_test_mm_lstm = pd.read_csv('C:/Users/mark_/userdata/Output/y_test_lstm_mm.csv', sep=',', decimal='.', header=None, index_col=False)#, axis=1)  # True unshuffled data. tflearn results = y_test_mm_lstm.csv 
    yhat_test_mm_lstm = pd.read_csv('C:/Users/mark_/userdata/Output/Yhat_test_lstm_mm.csv', sep=',', decimal='.', header=None, index_col=False)
    # Yhat_test_mm_4b = np.mean(pd.read_csv('C:/Users/mark_/userdata/Output/Yhat_v_mlp_4d_do5.csv', sep=',', decimal='.', header=None), axis=1)
    
    # Yhat_test_mm_4a = np.mean(pd.read_csv('C:/Users/mark_/userdata/Output/Yhat_regression_loo_100.csv', sep=',', decimal='.', header=None), axis=1) # 4th degree polynomial
    # Yhat_test_all_1l_b = np.mean(pd.read_csv('C:/Users/mark_/userdata/Output/Yhat_test_all_b.csv', sep=',', decimal='.', header=None), axis=1)  #b, shuffled data, 1-layer mlp
    # Yhat_test_all_ns_c = np.mean(pd.read_csv('C:/Users/mark_/userdata/Output/Yhat_test_all_c.csv', sep=',', decimal='.', header=None), axis=1)  ## c, not shuffled data, 6-layer mlp
    # Yhat_test_all_6l_d = np.mean(pd.read_csv('C:/Users/mark_/userdata/Output/Yhat_test_all_d.csv', sep=',', decimal='.', header=None), axis=1) # d, shuffled data, 6-layer mlp
    # Yhat_test_all_3l_e = np.mean(pd.read_csv('C:/Users/mark_/userdata/Output/Yhat_test_all_e.csv', sep=',', decimal='.', header=None), axis=1)# e, shuffled data, 3-layer mlp
    # Yhat_test_all_2l_f = np.mean(pd.read_csv('C:/Users/mark_/userdata/Output/Yhat_test_all_f.csv', sep=',', decimal='.', header=None), axis=1)# f, shuffled data, 2-layer mlp
    # Yhat_test_all_12l_g = np.mean(pd.read_csv('C:/Users/mark_/userdata/Output/Yhat_test_all_g.csv', sep=',', decimal='.', header=None), axis=1)# g, shuffled data, 12-layer mlp
    # Yhat_test_all_1l12n_h = np.mean(pd.read_csv('C:/Users/mark_/userdata/Output/Yhat_test_all_h.csv', sep=',', decimal='.', header=None), axis=1)# h, shuffled data, 1-layer mlp, 12 nodes
    
    
    Yhat_test_mm_lstm = np.mean(pd.read_csv('C:/Users/mark_/userdata/Output/Yhat_v_k_lstm_mm.csv', sep=',', decimal='.', header=None), axis=1)  # True unshuffled data. tflearn results = Yhat_test_lstm.csv 
    # Yhat_test_mm_5a = np.mean(pd.read_csv('C:/Users/mark_/userdata/Output/Yhat_v_mlp_5a.csv', sep=',', decimal='.', header=None), axis=1)
    # Yhat_test_mm_5b = np.mean(pd.read_csv('C:/Users/mark_/userdata/Output/Yhat_v_mlp_5b.csv', sep=',', decimal='.', header=None), axis=1)
    # Yhat_test_mm_5c = np.mean(pd.read_csv('C:/Users/mark_/userdata/Output/Yhat_v_mlp_5c.csv', sep=',', decimal='.', header=None), axis=1)
    # Yhat_test_mm_5d = np.mean(pd.read_csv('C:/Users/mark_/userdata/Output/Yhat_v_mlp_5d.csv', sep=',', decimal='.', header=None), axis=1)
    # Yhat_test_mm_5e = np.mean(pd.read_csv('C:/Users/mark_/userdata/Output/Yhat_v_mlp_5e.csv', sep=',', decimal='.', header=None), axis=1)
    # Yhat_test_mm_5f = np.mean(pd.read_csv('C:/Users/mark_/userdata/Output/Yhat_v_mlp_5f.csv', sep=',', decimal='.', header=None), axis=1)
 
##### F2_regression_abcd.   Plot predicted versus true value  ########
    # Yhat_test_mm_4h = np.mean(pd.read_csv('C:/Users/mark_/userdata/Output/Yhat_v_mlp_4h.csv', sep=',', decimal='.', header=None), axis=1)
    fig, ax = plt.subplots()
    # label = ['Experiment', 'Simulation']
    mk = 5
    ax.plot(xl,yl,  c="k")
    # ax.scatter(y_test_mm, Yhat_test_mm_4a, s=mk, c='b', marker='v')# 4th degree polynomial
    # ax.scatter(y_test_mm, Yhat_test_all_1l_b, s=mk, c='r', marker='D')
    # ax.scatter(y_test_mm_c, Yhat_test_all_ns_c, s=mk, c='y', marker='s')  # run c = not shuffled data
    ax.scatter(y_test_mm_t, Yhat_v, s=mk, c='k', marker='+')

    ax.set(xlabel='True values', ylabel='Predictions')
    # fig.savefig('C:/Users/mark_/Documents/02_Professional/03_Unil/SATW/MDPI_paper/MDPI_article/MDPI_article_v15/figures/F2_regression_abcd.svg', format="svg")
   
    plt.show()

##### F3_regression_defgh.   Plot predicted versus true value  ########
    # fig, ax = plt.subplots()
    # mk = 5
    # ax.plot(xl,yl,  c="k")
    # ax.scatter(y_test_mm, Yhat_test_all_2l_f, s=mk, c='g', marker='v')
    # ax.scatter(y_test_mm, Yhat_test_all_3l_e, s=mk, c='b', marker='v')
    # ax.scatter(y_test_mm, Yhat_test_all_6l_d, s=mk, c='r', marker='+')
    # ax.scatter(y_test_mm, Yhat_test_all_12l_g, s=mk, c='c', marker='v')
    # ax.scatter(y_test_mm, Yhat_test_all_1l12n_h, s=mk, c='y', marker='v')
    # # ax.scatter(y_test_mm, Yhat_test_mm_4e, s=mk, c='m', marker='s')
    # ax.set(xlabel='True values', ylabel='Predictions')
    # # fig.savefig('C:/Users/mark_/Documents/02_Professional/03_Unil/SATW/MDPI_paper/MDPI_article/MDPI_article_v15/figures/F3_regression_abcde.svg', format="svg")
   
    # plt.show()
    
    rms = sqrt(mean_squared_error(y_test_mm_c, np.asarray(Yhat_test_all_ns_c)))
    print(f"2a, Validation test rms: {100*rms:.2f} ")# " + str(rms))


    reg = LinearRegression(fit_intercept=False).fit(y_test_mm_t, np.asarray(Yhat_v)) #, fit_intercept=False forces intercept to zero, Yhat_polynomial
    print("log trans, Validation test R2: %.4f" % reg.score(y_test_mm_t, np.asarray(Yhat_v)))
    
    reg = LinearRegression(fit_intercept=False).fit(y_test_mm_c, np.asarray(Yhat_test_all_ns_c)) #, fit_intercept=False forces intercept to zero, Yhat_polynomial
    print("2a, Validation test R2: %.4f" % reg.score(y_test_mm_c, np.asarray(Yhat_test_all_ns_c)))
    
    
    # q = np.asarray(y_test_mm_lstm.to_numpy())
    # r = Yhat_test_mm_lstm.to_numpy().reshape(-1,1)
    # print(Yhat_test_mm_lstm)
    # print(y_test_mm_lstm.to_numpy())
    # print(np.asarray(Yhat_test_mm_lstm))
    
    ## LSTM model regression  ###
    # Fit intercept set to True. If set to False, the R2 is -2.7!!!
    reg = LinearRegression(fit_intercept=True).fit(np.asarray(y_test_mm_lstm.to_numpy()), Yhat_test_mm_lstm.to_numpy().reshape(-1,1)) #, fit_intercept=False forces intercept to zero, Yhat_polynomial
    print("2a, Validation test R2: %.4f" % reg.score(np.asarray(y_test_mm_lstm.to_numpy()), Yhat_test_mm_lstm.to_numpy().reshape(-1,1)))

    ## from math import sqrt
    testScore = sqrt(mean_squared_error(outputs_test[:,0], testPredict[:,0]))
    print('Test Score: %.2f RMSE' % (testScore))
    testScore = sqrt(mean_squared_error(np.asarray(y_test_mm_lstm.to_numpy()), Yhat_test_mm_lstm.to_numpy().reshape(-1,1)))
    print('Test Score: %.3f RMSE' % (testScore))
    MAE = mean_absolute_error(np.asarray(y_test_mm_lstm.to_numpy()), Yhat_test_mm_lstm.to_numpy().reshape(-1,1), sample_weight=None, multioutput='uniform_average')
    # MAE = mean_absolute_error(outputs_test[:,0], testPredict[:,0], sample_weight=None, multioutput='uniform_average')
    print('Mean absolute error (MAE): %.3f' % (MAE))
     
    time = np.arange(1, 460, 1) #(x for x in len(outputs_test))
    plt.scatter(time, np.asarray(y_test_mm_lstm.to_numpy()))
    plt.scatter(time, Yhat_test_mm_lstm.to_numpy().reshape(-1,1))
    
   # plt.plot(testPredict[:,0], outputs_test[:,0])
    # plt.plot(testPredict[:,0], outputs_test[:,0])
    # plt.scatter(time, testPredict[:,0])
    # plt.scatter(time, outputs_test[:,0])
    
    

    F2_AB, ((axa, axb)) = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(10,6)) #, (axg, axhsharex=True, sharey=True, 
    # F2_AB.tight_layout()
    # fig.text(0.5, 0.04, 'True values', ha='center')
    # fig.text(0.04, 0.5, 'Predictions', va='center', rotation='vertical')
    # plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.xlabel("True values")
    plt.ylabel("Predictions")
    mk = 6
    axa.scatter(y_test_mm, Yhat_test_mm_4a, s=mk, c='b', marker='v')
    axa.scatter(y_test_mm, Yhat_test_mm_4b, s=mk, c='r', marker='D')
    axa.scatter(y_test_mm_4f, Yhat_test_mm_4f, s=mk, c='y', marker='s')
    axa.scatter(y_test_mm_lstm, Yhat_test_mm_lstm, s=mk, c='k', marker='x')
    axa.plot(xl,yl,  c="k")
    # axa.set_title('2a) polynomial', fontsize=fs) #4th degree polynomial
    axb.scatter(y_test_mm, Yhat_test_mm_4c, s=mk, c='g', marker='v')
    axb.scatter(y_test_mm, Yhat_test_mm_4d, s=mk, c='c', marker='+')
    axb.scatter(y_test_mm, Yhat_test_mm_4e, s=mk, c='m', marker='s')
    axb.plot(xl,yl,  c="k")
    # F2_AB.subplots_adjust(hspace=0.4)
    # plt.savefig('C:/Users/mark_/Documents/02_Professional/03_Unil/SATW/MDPI_paper/MDPI_article/MDPI_article_v12/figures/F2_regression_1x2.svg', format="svg")
    plt.show()
    
# # # plt.savefig('C:/Users/mark_/Documents/02_Professional/03_Unil/SATW/MDPI_paper/MDPI_article/MDPI_article_v11/figures/F4_ABCD.svg', format="svg") #F4_ABCD
# # plt.savefig('/scratch/mmccormi1/F4_ABCD.svg', format="svg")

##### MDPI figure 2a - 4th order polynomial   ########
# To make this figure do preprocessing separately.
# plt.figure(figsize=(7,6))  
     
    # axa.scatter( y_test_mm, Yhat_test_mm_4a, linestyle='-', c="k", s=1)
    # ax.scatter(y_test_mm, Yhat_test_mm_4b, s=mk, c='r', marker='D')
    # axa.plot(xl,yl,  c="k")
    # axa.set_title('2a) polynomial', fontsize=fs) #4th degree polynomial



###### stats lstm ########

    rms = sqrt(mean_squared_error(y_test_mm_lstm, Yhat_test_mm_lstm))
    print(f"lstm, Validation test rms: {100*rms:.2f} ")# " + str(rms))

    reg = LinearRegression(fit_intercept=False).fit(y_test_mm_lstm, Yhat_test_mm_lstm) #Set fit_intercept to False to force 0. , normalize=False
    print("lstm, validation test R2: %.2f" % reg.score(y_test_mm_lstm, Yhat_test_mm_lstm))

    
# # # Manually create arrays for use in figure 4

# # y_test_mm_4a = y_test_mm
# # Yhat_v_4a = np.mean(Yhat_v_rep, axis = 0)

# # # y_test_mm_4b = y_test_mm
# # # Yhat_v_4b = np.mean(Yhat_v_rep, axis = 0)

# y_test_mm_4c = y_test_mm
# Yhat_v_4c =  np.mean(Yhat_v_rep, axis = 0)

# # # y_test_mm_4d = y_test_mm
# # # Yhat_v_4d = np.mean(Yhat_v_rep, axis = 0)

# plt.scatter(Yhat_test_mm_4g, y_test_mm_lstm, linestyle='-', c="k", s=1)

    F2_ABCDEF, ((axa, axb), (axc, axd), (axe, axf)) = plt.subplots(3, 2, figsize=(5,5)) #, (axg, axhsharex=True, sharey=True, 
    F2_ABCDEF.tight_layout()
    fs = 10
# # # plt.savefig('C:/Users/mark_/Documents/02_Professional/03_Unil/SATW/MDPI_paper/MDPI_article/MDPI_article_v11/figures/F4_ABCD.svg', format="svg") #F4_ABCD
# # plt.savefig('/scratch/mmccormi1/F4_ABCD.svg', format="svg")

##### MDPI figure 2a - 4th order polynomial   ########
# To make this figure do preprocessing separately.
# plt.figure(figsize=(7,6))  
     
    axa.scatter( y_test_mm, Yhat_test_mm_4a, linestyle='-', c="k", s=1)
    axa.plot(xl,yl,  c="k")
    axa.set_title('2a) polynomial', fontsize=fs) #4th degree polynomial
# plt.legend()#['True outputs (Test data)', 'Predictions (Test data)', 'Predictions (Test-2 data)'])
    # plt.xlabel('True values') #Measurement sequence number
    # plt.ylabel('Predictions')
# plt.title('MLP model - preshuffled data, gradual reduction in nimber of hidden layers, predicted and true values, validation test  Daily CV reduction (MinMax scaled)', fontsize='large') #MLP predicted and observed flow rate during testing
# plt.savefig('C:/Users/mark_/Documents/02_Professional/03_Unil/SATW/MDPI_paper/figures/F4_A_polynomial.svg', format="svg")
# plt.show()

    
### MDPI figure 2b  (1-layer NN)    #######
# # plt.figure(figsize=(7,6)) 
# plt.subplot(221)
    axb.scatter(y_test_mm,Yhat_test_mm_4b,  c="k", s=1)
# # plt.plot(y_pred_v)
    axb.plot(xl,yl,  c="k", linestyle='solid') #dashed
    axb.set_title('2b) 1-layer mlp', fontsize=fs) # 1-layer perceptron, \n 192 nodes'
# axa.annotate("R2: %.2f" % reg.score(Yhat_v_4a, y_test_mm_4a), (10,10))
# # plt.legend()#['True outputs (Test data)', 'Predictions (Test data)', 'Predictions (Test-2 data)'])
    # plt.xlabel('True values') #Measurement sequence number
    # plt.ylabel('Predictions')
    # plt.title('l-layer perceptron model - predicted and true values, validation test, daily CV reduction (MinMax scaled)', fontsize=8) #MLP predicted and observed flow rate during testing.  'large'
# # plt.savefig('C:/Users/mark_/Documents/02_Professional/03_Unil/SATW/MDPI_paper/MDPI_article/MDPI_article_v09/figures/F4_A_1-layer_test_shuffled.svg', format="svg")
    # plt.show() 
    rms = sqrt(mean_squared_error(y_test_mm, Yhat_test_mm_4b))
    print(f"2b, Validation test rms: {100*rms:.2f} ")# + str(rms))

    reg = LinearRegression(fit_intercept=False).fit(y_test_mm, Yhat_test_mm_4b) #Set fit_intercept to False to force 0. , normalize=False
    print("2b, validation test R2: %.2f" % reg.score(y_test_mm, Yhat_test_mm_4b))


#### MDPI figure 2c - 2-layer MLP (equal number of nodes per layer) with unshuffled data    ##########
# To make this figure do preprocessing separately.
# Yhat_v is returned from the satw_mlp_v5 file
# plt.figure(figsize=(7,6))  
# plt.subplot(222) 
    axc.scatter(y_test_mm, Yhat_test_mm_4c, linestyle='-',  c="k", s=1)
    axc.plot(xl,yl,  c="k")
    axc.set_title('2c) 2-layer mlp', fontsize=fs)
# # plt.legend()#['True outputs (Test data)', 'Predictions (Test data)', 'Predictions (Test-2 data)'])
    # plt.xlabel('True values') #Measurement sequence number
    # plt.ylabel('Predictions')
# plt.title('MLP model - unshuffled data, predicted and true values, validation test  Daily CV reduction (MinMax scaled)', fontsize=8) #MLP predicted and observed flow rate during testing
# # plt.savefig('C:/Users/mark_/Documents/02_Professional/03_Unil/SATW/MDPI_paper/MDPI_article/MDPI_article_v09/figures/F4_B_6-layer-MLP_test_unshuffled.svg', format="svg")
# plt.show()

    rms = sqrt(mean_squared_error(y_test_mm, Yhat_test_mm_4c))
    print(f"2c, Validation test rms: {100*rms:.2f} ")# " + str(rms))

    reg = LinearRegression(fit_intercept=False).fit(y_test_mm, Yhat_test_mm_4c) #Use y_test_mm_4b because shuffled data. Set fit_intercept to False to force 0. , normalize=False
    print("2c, Validation test R2: %.2f" % reg.score(y_test_mm, Yhat_test_mm_4c))


##### MDPI figure 2d - 3-layer MLP (equal number of nodes per layer) with PRESHUFFLED data    ##########
# To make this figure do preprocessing separately.
# Yhat_v is returned from the satw_mlp_v5 file
#plt.figure(figsize=(7,6))   
    axd.scatter(y_test_mm, Yhat_test_mm_4d, linestyle='-', c="k", s=1)
    axd.plot(xl,yl,  c="k")
    axd.set_title('2d) 3-layer mlp', fontsize=fs) #3-layer perceptron \n 192 nodes per layer, \n shuffled'
# plt.legend()#['True outputs (Test data)', 'Predictions (Test data)', 'Predictions (Test-2 data)'])
    # plt.xlabel('True values') #Measurement sequence number
    # plt.ylabel('Predictions')
# plt.title('MLP model - preshuffled data, predicted and true values, validation test  Daily CV reduction (MinMax scaled)', fontsize='large') #MLP predicted and observed flow rate during testing
# # plt.savefig('C:/Users/mark_/Documents/02_Professional/03_Unil/SATW/MDPI_paper/MDPI_article/MDPI_article_v09/figures/F4_C_6-layer-MLP_test_preshuffled.svg', format="svg")
# plt.show()
    rms = sqrt(mean_squared_error(y_test_mm, Yhat_test_mm_4d))
    print(f"2d, Validation test rms: {100*rms:.2f} ")# " + str(rms))

    reg = sklearn.linear_model.LinearRegression(fit_intercept=False).fit(y_test_mm, Yhat_test_mm_4d) # Set fit_intercept to False to force 0. , normalize=False
    print("2d, Validation test R2: %.2f" % reg.score(y_test_mm, Yhat_test_mm_4d))

##### MDPI figure 2e - 6-layer MLP (equal number of nodes per layer) with PRESHUFFLED data 
# To make this figure do preprocessing separately.
# Yhat_v is returned from the satw_mlp_v5 file
#plt.figure(figsize=(7,6))   
    axe.scatter(y_test_mm, Yhat_test_mm_4e, linestyle='-', c="k", s=1)
    axe.plot(xl,yl,  c="k")
    axe.set_title('2e) 6-layer mlp', fontsize=fs) #6-layer perceptron, shuffled
# plt.legend()#['True outputs (Test data)', 'Predictions (Test data)', 'Predictions (Test-2 data)'])
    # plt.xlabel('True values') #Measurement sequence number
    # plt.ylabel('Predictions')
# plt.title('MLP model - preshuffled data, predicted and true values, validation test  Daily CV reduction (MinMax scaled)', fontsize='large') #MLP predicted and observed flow rate during testing
# # plt.savefig('C:/Users/mark_/Documents/02_Professional/03_Unil/SATW/MDPI_paper/MDPI_article/MDPI_article_v09/figures/F4_C_6-layer-MLP_test_preshuffled.svg', format="svg")
# plt.show()
    rms = sqrt(mean_squared_error(y_test_mm, Yhat_test_mm_4e))
    print(f"2e, Validation test rms: {100*rms:.2f} ")# " + str(rms))

    reg = sklearn.linear_model.LinearRegression(fit_intercept=False).fit(y_test_mm, Yhat_test_mm_4e) # Set fit_intercept to False to force 0. , normalize=False
    print("2e, Validation test R2: %.2f" % reg.score(y_test_mm, Yhat_test_mm_4e))

##### MDPI figure 4f - 6-layer MLP (equal number of nodes per layer) with NOT SHUFFLED data   ########
# To make this figure do preprocessing separately.
# Yhat_v is returned from the satw_mlp_v5 file
# plt.figure(figsize=(7,6))   
    axf.scatter(y_test_mm_4f, Yhat_test_mm_4f, linestyle='-', c="k", s=1)
    axf.plot(xl,yl,  c="k")
    axf.set_title('2f) 6-layer mlp, not shuffled', fontsize=fs)#11-layer MLP, \n grad inc and red in \n num units per layer, shuffled6-layer perceptron,\n rectangle, NOT shuffled
# plt.legend()#['True outputs (Test data)', 'Predictions (Test data)', 'Predictions (Test-2 data)'])
    plt.xlabel('True values') #Measurement sequence number
    plt.ylabel('Predictions')
# plt.title('MLP model - preshuffled data, gradual reduction in nimber of hidden layers, predicted and true values, validation test  Daily CV reduction (MinMax scaled)', fontsize='large') #MLP predicted and observed flow rate during testing
# plt.savefig('C:/Users/mark_/Documents/02_Professional/03_Unil/SATW/MDPI_paper/MDPI_article/MDPI_article_v09/figures/F4_D_6-layer-MLP_test_preshuffled_gradual.svg', format="svg")
# plt.show()
    rms = sqrt(mean_squared_error(y_test_mm_4f, Yhat_test_mm_4f))
    print(f"2f, Validation test rms: {100*rms:.2f} ")# " + str(rms))

    reg = LinearRegression(fit_intercept=False).fit(y_test_mm_4f, Yhat_test_mm_4f) #, fit_intercept=True
    print("2f, Validation test R2: %.2f" % reg.score(y_test_mm_4f, Yhat_test_mm_4f))

    F2_ABCDEF.subplots_adjust(hspace=0.4)
    plt.savefig('C:/Users/mark_/Documents/02_Professional/03_Unil/SATW/MDPI_paper/MDPI_article/MDPI_article_v12/figures/F2_regression_abcdef.svg', format="svg")
    plt.show()
    

#     F5_ABCDEF, ((axa, axb), (axc, axd), (axe, axf)) = plt.subplots(3, 2, figsize=(4,4)) #sharex=True, sharey=True, 
#     F5_ABCDEF.tight_layout()
# # # # plt.savefig('C:/Users/mark_/Documents/02_Professional/03_Unil/SATW/MDPI_paper/MDPI_article/MDPI_article_v11/figures/F4_ABCD.svg', format="svg") #F4_ABCD
# # # plt.savefig('/scratch/mmccormi1/F4_ABCD.svg', format="svg")





###############################   ARCHIVES    ####################################################

# ##### MDPI figure 5a -    ########
# # To make this figure do preprocessing separately.
# # plt.figure(figsize=(7,6))  
     
#     axa.scatter(y_test_mm, Yhat_test_mm_5a, linestyle='-', c="k", s=1)
#     axa.plot(xl,yl,  c="k")
#     axa.set_title('5a 3-layer perceptron \n 32 nodes per layer \n shuffled', fontsize=8)
# # plt.legend()#['True outputs (Test data)', 'Predictions (Test data)', 'Predictions (Test-2 data)'])
#     plt.xlabel('True values') #Measurement sequence number
#     plt.ylabel('Predictions')
# # plt.title('MLP model - preshuffled data, gradual reduction in nimber of hidden layers, predicted and true values, validation test  Daily CV reduction (MinMax scaled)', fontsize='large') #MLP predicted and observed flow rate during testing
# # plt.savefig('C:/Users/mark_/Documents/02_Professional/03_Unil/SATW/MDPI_paper/figures/F4_A_polynomial.svg', format="svg")
# # plt.show()
#     rms = sqrt(mean_squared_error(y_test_mm, Yhat_test_mm_5a))
#     print(f"5a, Validation test rms: {100*rms:.2f} ")# " + str(rms))

#     reg = LinearRegression(fit_intercept=False).fit(y_test_mm, Yhat_test_mm_5a) #, fit_intercept=False forces intercept to zero, Yhat_polynomial
#     print("5a, Validation test R2: %.2f" % reg.score(y_test_mm, Yhat_test_mm_5a))
    
# ### MDPI figure 5b      #######
# # # plt.figure(figsize=(7,6)) 
# # plt.subplot(221)
#     axb.scatter(y_test_mm, Yhat_test_mm_5b, c="k", s=1)
# # # plt.plot(y_pred_v)
#     axb.plot(xl,yl,  c="k", linestyle='solid') #dashed
#     axb.set_title('5b 3-layer perceptron \n 192 nodes per layer \n shuffled', fontsize=8)
# # axa.annotate("R2: %.2f" % reg.score(Yhat_v_4a, y_test_mm_4a), (10,10))
# # # plt.legend()#['True outputs (Test data)', 'Predictions (Test data)', 'Predictions (Test-2 data)'])
#     plt.xlabel('True values') #Measurement sequence number
#     plt.ylabel('Predictions')
#     # plt.title('l-layer perceptron model - predicted and true values, validation test, daily CV reduction (MinMax scaled)', fontsize=8) #MLP predicted and observed flow rate during testing.  'large'
# # # plt.savefig('C:/Users/mark_/Documents/02_Professional/03_Unil/SATW/MDPI_paper/MDPI_article/MDPI_article_v09/figures/F4_A_1-layer_test_shuffled.svg', format="svg")
#     # plt.show() 
#     rms = sqrt(mean_squared_error(y_test_mm, Yhat_test_mm_5b))
#     print(f"5b, Validation test rms: {100*rms:.2f} ")# " + str(rms))

#     reg = LinearRegression(fit_intercept=False).fit(y_test_mm, Yhat_test_mm_5b) #Set fit_intercept to False to force 0. , normalize=False
#     print("5b, validation test R2: %.2f" % reg.score(y_test_mm, Yhat_test_mm_5b))


# #### MDPI figure 5c -3-layer MLP (equal number of nodes per layer) with unshuffled data    ##########
# # To make this figure do preprocessing separately.
# # Yhat_v is returned from the satw_mlp_v5 file
# # plt.figure(figsize=(7,6))  
# # plt.subplot(222) 
#     axc.scatter(y_test_mm, Yhat_test_mm_5c, linestyle='-',  c="k", s=1)
#     axc.plot(xl,yl,  c="k")
#     axc.set_title('5c 3-layer perceptron,\n 640 nodes per layer, shuffled', fontsize=8)
# # # plt.legend()#['True outputs (Test data)', 'Predictions (Test data)', 'Predictions (Test-2 data)'])
#     plt.xlabel('True values') #Measurement sequence number
#     plt.ylabel('Predictions')
# # plt.title('MLP model - unshuffled data, predicted and true values, validation test  Daily CV reduction (MinMax scaled)', fontsize=8) #MLP predicted and observed flow rate during testing
# # # plt.savefig('C:/Users/mark_/Documents/02_Professional/03_Unil/SATW/MDPI_paper/MDPI_article/MDPI_article_v09/figures/F4_B_6-layer-MLP_test_unshuffled.svg', format="svg")
# # plt.show()
#     rms = sqrt(mean_squared_error(y_test_mm, Yhat_test_mm_5c))
#     print(f"5c, Validation test rms: {100*rms:.2f} ")# " + str(rms))

#     reg = LinearRegression(fit_intercept=False).fit(y_test_mm, Yhat_test_mm_5c) #Use y_test_mm_4b because shuffled data. Set fit_intercept to False to force 0. , normalize=False
#     print("5c, Validation test R2: %.2f" % reg.score(y_test_mm, Yhat_test_mm_5c))


# ##### MDPI figure 5d - 3-layer MLP  with PRESHUFFLED data    ##########
# # To make this figure do preprocessing separately.
# # Yhat_v is returned from the satw_mlp_v5 file
# #plt.figure(figsize=(7,6))   
#     axd.scatter(Yhat_test_mm_5d, y_test_mm, linestyle='-', c="k", s=1)
#     axd.plot(xl,yl,  c="k")
#     axd.set_title('5d 3-layer perceptron \n 1280 nodes per layer, \n shuffled', fontsize=8)
# # plt.legend()#['True outputs (Test data)', 'Predictions (Test data)', 'Predictions (Test-2 data)'])
#     plt.xlabel('True values') #Measurement sequence number
#     plt.ylabel('Predictions')
# # plt.title('MLP model - preshuffled data, predicted and true values, validation test  Daily CV reduction (MinMax scaled)', fontsize='large') #MLP predicted and observed flow rate during testing
# # # plt.savefig('C:/Users/mark_/Documents/02_Professional/03_Unil/SATW/MDPI_paper/MDPI_article/MDPI_article_v09/figures/F4_C_6-layer-MLP_test_preshuffled.svg', format="svg")
# # plt.show()
#     rms = sqrt(mean_squared_error(y_test_mm, Yhat_test_mm_5d))
#     print(f"5d, Validation test rms: {100*rms:.2f} ")# " + str(rms))

#     reg = sklearn.linear_model.LinearRegression(fit_intercept=False).fit(y_test_mm, Yhat_test_mm_5d) # Set fit_intercept to False to force 0. , normalize=False
#     print("5d, Validation test R2: %.2f" % reg.score(y_test_mm, Yhat_test_mm_5d))

# ##### MDPI figure 5e - MLP , 12-layer, 192 nodes per layer, with preshuffled data
# # To make this figure do preprocessing separately.
# # Yhat_v is returned from the satw_mlp_v5 file
# #plt.figure(figsize=(7,6))   
#     axe.scatter(Yhat_test_mm_5e, y_test_mm, linestyle='-', c="k", s=1)
#     axe.plot(xl,yl,  c="k")
#     axe.set_title('5e 12-layer perceptron,\n 192 nodes per layer, shuffled', fontsize=8)
# # plt.legend()#['True outputs (Test data)', 'Predictions (Test data)', 'Predictions (Test-2 data)'])
#     plt.xlabel('True values') #Measurement sequence number
#     plt.ylabel('Predictions')
# # plt.title('MLP model - preshuffled data, predicted and true values, validation test  Daily CV reduction (MinMax scaled)', fontsize='large') #MLP predicted and observed flow rate during testing
# # # plt.savefig('C:/Users/mark_/Documents/02_Professional/03_Unil/SATW/MDPI_paper/MDPI_article/MDPI_article_v09/figures/F4_C_6-layer-MLP_test_preshuffled.svg', format="svg")
# # plt.show()
#     rms = sqrt(mean_squared_error(y_test_mm, Yhat_test_mm_5e))
#     print(f"5e, Validation test rms: {100*rms:.2f} ")# " + str(rms))

#     reg = sklearn.linear_model.LinearRegression(fit_intercept=False).fit(y_test_mm, Yhat_test_mm_5e) # Set fit_intercept to False to force 0. , normalize=False
#     print("5e, Validation test R2: %.2f" % reg.score(y_test_mm, Yhat_test_mm_5e))

# ##### MDPI figure 5f - 2-layer MLP, 192 nodes per layer \n with preshuffled data    ########
# # To make this figure do preprocessing separately.
# # Yhat_v is returned from the satw_mlp_v5 file
# # plt.figure(figsize=(7,6))   
#     axf.scatter(y_test_mm, Yhat_test_mm_5f, linestyle='-', c="k", s=1)
#     axf.plot(xl,yl,  c="k")
#     axf.set_title('5f 2-layer MLP, 192 nodes per layer, \n  shuffled', fontsize=8)
# # plt.legend()#['True outputs (Test data)', 'Predictions (Test data)', 'Predictions (Test-2 data)'])
#     plt.xlabel('True values') #Measurement sequence number
#     plt.ylabel('Predictions')
# # plt.title('MLP model - preshuffled data, gradual reduction in nimber of hidden layers, predicted and true values, validation test  Daily CV reduction (MinMax scaled)', fontsize='large') #MLP predicted and observed flow rate during testing
# # plt.savefig('C:/Users/mark_/Documents/02_Professional/03_Unil/SATW/MDPI_paper/MDPI_article/MDPI_article_v09/figures/F4_D_6-layer-MLP_test_preshuffled_gradual.svg', format="svg")
# # plt.show()
#     rms = sqrt(mean_squared_error(y_test_mm, Yhat_test_mm_5f))
#     print(f"5f, Validation test rms: {100*rms:.2f} ")# " + str(rms))


#     reg = LinearRegression(fit_intercept=False).fit(y_test_mm, Yhat_test_mm_5f) #, fit_intercept=True
#     print("4f, Validation test R2: %.2f" % reg.score(y_test_mm, Yhat_test_mm_5f))

#     F5_ABCDEF.subplots_adjust(hspace=0.9)
#     plt.show()    
# # # plt.savefig('C:/Users/mark_/Documents/02_Professional/03_Unil/SATW/MDPI_paper/MDPI_article/MDPI_article_v11/figures/F4_ABCD.svg', format="svg") #F4_ABCD

# ##### MDPI figure 4f - 3-layer MLP   ########
# # To make this figure do preprocessing separately.
# # plt.figure(figsize=(7,6))   
#     axe.scatter(Yhat_test_mm_4f, y_test_mm, linestyle='-', c="k", s=1)
#     axe.plot(xl,yl,  c="k")
#     axe.set_title('4f 3-layer mlp', fontsize=10)
# # plt.legend()#['True outputs (Test data)', 'Predictions (Test data)', 'Predictions (Test-2 data)'])
#     plt.xlabel('True values') #Measurement sequence number
#     plt.ylabel('Predictions')
# # plt.title('MLP model - preshuffled data, gradual reduction in nimber of hidden layers, predicted and true values, validation test  Daily CV reduction (MinMax scaled)', fontsize='large') #MLP predicted and observed flow rate during testing
# # plt.savefig('C:/Users/mark_/Documents/02_Professional/03_Unil/SATW/MDPI_paper/MDPI_article/MDPI_article_v09/figures/F4_D_6-layer-MLP_test_preshuffled_gradual.svg', format="svg")
# # plt.show()

#     reg = LinearRegression(fit_intercept=False).fit(y_test_mm, Yhat_test_mm_4f) #, fit_intercept=True
#     print("4e, Validation test R2: %.2f" % reg.score(y_test_mm, Yhat_test_mm_4f))


# # # plt.savefig('C:/Users/mark_/Documents/02_Professional/

############ Figures from the SATW intermediate report  ##############

# #Removal efficiency: 100*(TS*PCSin - TS*PCSout)/TS*PCSin
# InPCS = 10*np.multiply(Surr_data.iloc[:,[8]], Surr_data.iloc[:,[12]])  # 10 g/l per %
# OutPCS = 10*np.multiply(Surr_data.iloc[:,[56]], Surr_data.iloc[:,[60]])
# Elim = np.subtract(InPCS.iloc[:,[0]], OutPCS.iloc[:,[0]])
# #print(Elim.iloc[:,[0]])
# Ecap = np.multiply(Surr_data.iloc[:,[2]],Elim.iloc[:,[0]])/0.056 #vol AFBR = 56 litres
# plt.figure(figsize=(8,4.5))
# with plt.style.context(('ggplot')):
#     plt.plot(Date, Ecap)
#     plt.title('Elimination capacity in terms of calorific value reduction and reactor volume')
#     plt.xlabel('Date')
#     plt.xticks(rotation='vertical')
#     plt.ylabel('EC [kJ.litre$^-$$^1$.day$^-$$^1$]')
# plt.show()

# # Inlet flow
# Q_i = Surr_data.iloc[:,[2]]
# plt.figure(figsize=(8,4.5))
# with plt.style.context(('ggplot')):
#     plt.plot(Date, Q_i)
#     plt.title('Inlet flow rate')
#     plt.xlabel('Date')
#     plt.xticks(rotation='vertical')
#     plt.ylabel('Inlet flow rate [litres$^-$$^1$.day$^-$$^1$]')
# plt.show()
# plt.savefig('C://Figures/Qin.png')
# print('Comment: Almost all of the inlet flow rate values are measured values. Note the generally increasing flowrate during the study.')

# # Inlet temperature
# T_i = Surr_data.iloc[:,[4]]
# plt.figure(figsize=(8,4.5))
# with plt.style.context(('ggplot')):
#     plt.plot(Date, T_i)
#     plt.title('Inlet temperature')
#     plt.xlabel('Date')
#     plt.xticks(rotation='vertical')
#     plt.ylabel('Temperature [°C]')
# plt.show()
# plt.savefig('C://Figures/Tin.png')
# print('Comment: Almost all of the temperature values are measured values. Note the cold temperature at the end of the study.')

# # Linear flow rate
# V_l = (Surr_data.iloc[:,[2]]/24000)/0.363
# plt.figure(figsize=(8,4.5))
# with plt.style.context(('ggplot')):
#     plt.plot(Date, V_l)
#     plt.title('Linear flow rate')
#     plt.xlabel('Date')
#     plt.xticks(rotation='vertical')
#     plt.ylabel('Linear flow rate [meters.$^-$$^1$]')
# plt.show()
# plt.savefig('C://Figures/V_l.png')

# # Hydraulic Retention Time
# HRT = 24*56/Surr_data.iloc[:,[2]]
# plt.figure(figsize=(8,4.5))
# with plt.style.context(('ggplot')):
#     plt.plot(Date, HRT)
#     plt.title('Hydraulic retention time')
#     plt.xlabel('Date')
#     plt.xticks(rotation='vertical')
#     plt.ylabel('Hydraulic retention time [hours]')
# plt.show()
# plt.savefig('C://Figures/HRT.png')

# # Re, packed bed
# labels = ['small packing','medium packing','large packing']
# #Re_s = (V_l/3600)*1050*0.006/0.003
# Re_m = (V_l/3600)*1050*0.012/0.003
# #Re_l = (V_l/3600)*1050*0.036/0.003
# plt.figure(figsize=(8,4.5))
# with plt.style.context(('ggplot')):
#     #plt.plot(Date, Re_s)
#     plt.plot(Date, Re_m)
#     #plt.plot(Date, Re_l)
#     #plt.legend(labels)
#     plt.title('Re, packed bed Reynolds number')
#     plt.xlabel('Date')
#     plt.xticks(rotation='vertical')
#     plt.ylabel('Re')
# plt.show()
# plt.savefig('C://Figures/Re.png')
# print('Comment: Note that the Reynolds number is more than 1000 times less than the limit for turbulent flow.')

# # Biogas volume fraction
# fBG = 100*np.divide(Surr_data.iloc[:,[1]],Surr_data.iloc[:,[2]])
# plt.figure(figsize=(8,4.5))
# with plt.style.context(('ggplot')):
#     plt.plot(Date, fBG)
#     plt.title('Biogas volume as a fraction of liquid + gas flow')
#     plt.xlabel('Date')
#     plt.xticks(rotation='vertical')
#     plt.ylabel('Biogas volume fraction [%]')
# plt.show()
# plt.savefig('C://Figures/BGvf.png')
# print('Comment: Biogas in the anaerobic filter occupies less than 10% of the fluid volume.')

# #Shear rate [s-1]
# #Sr_s = (V_l/3600)/0.00145
# Sr_m = (V_l/3600)/0.0064
# #Sr_l = (V_l/3600)/0.0188
# plt.figure(figsize=(8,4.5))
# with plt.style.context(('ggplot')):
#     #plt.plot(Date, Sr_s)
#     plt.plot(Date, Sr_m)
#     #plt.plot(Date, Sr_l)
#     #plt.legend(labels)
#     plt.title('Liquid shear rate')
#     plt.xlabel('Date')
#     plt.xticks(rotation='vertical')
#     plt.ylabel('Liquid shear rate [s$^-$$^1$]')
# plt.show()
# plt.savefig('C://Figures/liqSR.png')
# print('Comment: The shear rate is very low. Up to 50 s-1 observed in upflow digesters (Jiang, 2014)')

# # Loading rate
# LR = np.multiply(Surr_data.iloc[:,[0]], Surr_data.iloc[:,[2]])/0.056
# plt.figure(figsize=(8,4.5))
# with plt.style.context(('ggplot')):
#     plt.plot(Date, LR)
#     plt.title('Inlet Loading rate in terms of solids calorific value')
#     plt.xlabel('Date')
#     plt.xticks(rotation='vertical')
#     plt.ylabel('Loading rate [kJ.litre$^-$$^1$.day$^-$$^1$]')
# plt.show()
# plt.savefig('C://Figures/LR.png')
# print('Comment: The loading rate was increased during the study.')

# ### Elimination capacity: Qin * (TS*PCSin - TS*PCSout)/0,056
# InPCS = 10*np.multiply(Surr_data.iloc[:,[8]], Surr_data.iloc[:,[12]])  # 10 g/l per %
# OutPCS = 10*np.multiply(Surr_data.iloc[:,[56]], Surr_data.iloc[:,[60]])
# Elim = np.subtract(InPCS.iloc[:,[0]], OutPCS.iloc[:,[0]])
# #print(Elim.iloc[:,[0]])
# Ecap = np.multiply(Surr_data.iloc[:,[2]],Elim.iloc[:,[0]])/0.056 #vol AFBR = 56 litres
# plt.figure(figsize=(8,4.5))
# with plt.style.context(('ggplot')):
#     plt.plot(Date, Ecap)
#     plt.title('Elimination capacity in terms of calorific value reduction and reactor volume')
#     plt.xlabel('Date')
#     plt.xticks(rotation='vertical')
#     plt.ylabel('EC [kJ.litre$^-$$^1$.day$^-$$^1$]')
# plt.show()
# plt.savefig('C://Figures/EC.png')
# print('Comment: The Elimination capacity improved during the study in spite of increased influent flow rate and decreasing water temperature')

# #### Scatter plots
# plt.figure(figsize=(8,4.5))
# with plt.style.context(('ggplot')):
#     plt.scatter(T_i, CH4, s = 30, color = '#539caf', alpha = 0.75)
#     plt.title('CH4 production vs Inlet temperature')
#     plt.xlabel('Inlet temperature [°C]')
#     #plt.xticks(rotation='vertical')
#     plt.ylabel('CH4 production [liters.day$^-$$^1$]')
# plt.show()
# plt.savefig('C://Figures/CH4_Ti.png')
# print('Comment: The CH4 production is independant of the water temperature (above 10°C)')

# plt.figure(figsize=(8,4.5))
# with plt.style.context(('ggplot')):
#     plt.scatter(T_i, Ecap, s = 30, color = '#539caf', alpha = 0.75)
#     plt.title('Elimination capacity vs Inlet temperature')
#     plt.xlabel('Inlet temperature [°C]')
#     #plt.xticks(rotation='vertical')
#     plt.ylabel('EC [kJ.litre$^-$$^1$.day$^-$$^1$]')
# plt.show()
# plt.savefig('C://Figures/CH4_Ti.png')
# print('Comment: The Elimination capacity is independent of influent water temperature')

# plt.figure(figsize=(8,4.5))
# with plt.style.context(('ggplot')):
#     plt.scatter(Re_m, CH4, s = 30, color = '#539caf', alpha = 0.75)
#     plt.title('CH4 production vs Packed bed Reynolds number')
#     plt.xlabel('Re, Reynolds number')
#     #plt.xticks(rotation='vertical')
#     plt.ylabel('CH4 production [liters.day$^-$$^1$]')
# plt.show()
# plt.savefig('C://Figures/CH4_Re.png')
# print('Comment: The CH4 production increases with the Reynolds number. This suggests that turbulence should be promoted.')

# plt.figure(figsize=(8,4.5))
# with plt.style.context(('ggplot')):
#     plt.scatter(Sr_m, CH4, s = 30, color = '#539caf', alpha = 0.75)
#     plt.title('CH4 production vs shear rate')
#     plt.xlabel('Liquid shear rate [s.$^-$$^1$]')
#     plt.xticks(rotation='vertical')
#     plt.xlim((0,0.003))
#     plt.ylabel('CH4 production [litres.day$^-$$^1$]')
# plt.show()
# plt.savefig('C://Figures/CH4_Sr.png')
# print('Comment: The CH4 production increases with the shear rate. This suggests that shearing was not a problem.')

# plt.figure(figsize=(8,4.5))
# with plt.style.context(('ggplot')):
#     plt.scatter(Sr_m, Ecap, s = 30, color = '#539caf', alpha = 0.75)
#     plt.title('Elimination capacity (EC) vs shear rate')
#     plt.xlabel('Liquid shear rate [s.day$1^-1$]')
#     plt.xticks(rotation='vertical')
#     plt.xlim((0,0.003))
#     plt.ylabel('EC [kJ.litre$^-$$^1$.day$^-$$^1$]')
# plt.show()
# plt.savefig('C://Figures/Ecap_Sr.png')
# print('Comment: The Elimination capacity increases (or is independent of) with the shear rate. This suggests that shearing was not a problem.')

# plt.figure(figsize=(8,4.5))
# with plt.style.context(('ggplot')):
#     plt.scatter(LR, Ecap, s = 30, color = '#539caf', alpha = 0.75)
#     plt.title('Loading rate vs Elimination capacity')
#     plt.xlabel('Loading rate [kJ.litre$^-$$^1$.day$^-$$^1$]')
#     plt.xticks(rotation='vertical')
#     plt.ylabel('EC [kJ.litre$^-$$^1$.day$^-$$^1$]')
# plt.show()
# plt.savefig('C://Figures/Ecap_LR.png')
# print('Comment: The Elimination capacity increases with the loading rate. This suggests that the AFBR was not overloaded.')

# plt.figure(figsize=(8,4.5))
# with plt.style.context(('ggplot')):
#     plt.scatter(Ecap, CH4, s = 30, color = '#539caf', alpha = 0.75)
#     plt.title('CH4 production vs Elimination capacity')
#     plt.xlabel('Elimination capacity [kJ.litre$^-$$^1$.day$^-$$^1$]')
#     plt.xticks(rotation='vertical')
#     plt.ylabel('CH4 production [litres.day$^-$$^1$]')
# plt.show()
# plt.savefig('C://Figures/Ecap_CH4.png')
# print('Comment: The CH4 production is positively correlated with the elimination capacity. Large dispersion might be due to variation in biogas CO2 content.')
# print()

# ### Exploratory
# print('DESCRIPTIONS BASED ON THE USE OF SURROGATE DATA. EXPLORATORY PURPOSE')
# # Re, packed bed
# labels = ['small packing','medium packing','large packing']
# Re_s = (V_l/3600)*1050*0.006/0.003
# Re_m = (V_l/3600)*1050*0.012/0.003
# Re_l = (V_l/3600)*1050*0.036/0.003
# plt.figure(figsize=(8,4.5))
# with plt.style.context(('ggplot')):
#     plt.plot(Date, Re_s)
#     plt.plot(Date, Re_m)
#     plt.plot(Date, Re_l)
#     plt.legend(labels)
#     plt.title('Re, packed bed Reynolds number')
#     plt.xlabel('Date')
#     plt.xticks(rotation='vertical')
#     plt.ylabel('Re')
# plt.show()
# plt.savefig('C://Figures/Re.png')
# print('Comment: The predicted effect of packing size on the Reynolds number.')

# #Shear rate [s-1]
# Sr_s = (V_l/3600)/0.00145
# Sr_m = (V_l/3600)/0.0064
# Sr_l = (V_l/3600)/0.0188
# plt.figure(figsize=(8,4.5))
# with plt.style.context(('ggplot')):
#     plt.plot(Date, Sr_s)
#     plt.plot(Date, Sr_m)
#     plt.plot(Date, Sr_l)
#     plt.legend(labels)
#     plt.title('Liquid shear rate')
#     plt.xlabel('Date')
#     plt.xticks(rotation='vertical')
#     plt.ylabel('Liquid shear rate [s$^-$$^1$]')
# plt.show()
# plt.savefig('C://Figures/liqSR.png')
# print('Comment: The predicted effect of packing size on the Reynolds number.')

# ##Plot all surrogate CH4 production data
# Experiment = pd.read_excel('C:\\Users\\mark_\\Anaconda3\Data\SATW_project.xlsx', sheet_name='Exper_data', sep=";", decimal=',')  
# ExDF = pd.DataFrame(Experiment)
# ExE = Experiment.iloc[:,5]
# ExH = Experiment.iloc[:,7]#'HRT [h], CH4 [l/day]'
# ExH1 = Experiment.iloc[:,12]
# Ex1 = Experiment.iloc[:,13]
# ExH2 = Experiment.iloc[:,15]
# Ex2 = Experiment.iloc[:,16]
# ExH3 = Experiment.iloc[:,18]
# Ex3 = Experiment.iloc[:,19]
# ExH4 = Experiment.iloc[:,21]
# Ex4 = Experiment.iloc[:,22]
# ExH5 = Experiment.iloc[:,24]
# Ex5 = Experiment.iloc[:,25]
# ExH6 = Experiment.iloc[:,27]
# Ex6 = Experiment.iloc[:,28]
# ExH7 = Experiment.iloc[:,30]
# Ex7 = Experiment.iloc[:,31]
# ExH8 = Experiment.iloc[:,33]
# Ex8 = Experiment.iloc[:,34]
# ExH9 = Experiment.iloc[:,36]
# Ex9 = Experiment.iloc[:,37]


# plt.figure(figsize=(8,4.5))
# with plt.style.context(('ggplot')):
#     plt.scatter(ExH,ExE)
#     plt.scatter(ExH1,Ex1)
#     plt.scatter(ExH2,Ex2)
#     plt.scatter(ExH3,Ex3)
#     plt.scatter(ExH4,Ex4)
#     plt.scatter(ExH5,Ex5)
#     plt.scatter(ExH6,Ex6)
#     plt.scatter(ExH7,Ex7)
#     plt.scatter(ExH8,Ex8)
#     plt.scatter(ExH9,Ex9)
#     plt.legend()
#     plt.title('Methane production vs HRT (surrogate data)')
#     plt.xlabel('Hydraulic Retention Time (HRT) [hours]')
#     #plt.xticks(rotation='vertical')
#     plt.ylabel('Methane production [l/day]')
# plt.show()
# print('Comment: The swarm of response data points that will be used to build the DNN model. (Approximated 227*9 = 2043 points')
#%%
# ### Create L9 NN training data array 
    RD = pd.read_excel('C:\\Users\\mark_\\Anaconda3\Data\SATW_project.xlsx', sheet_name='Database', sep=";", decimal=',')  
    RD_Ex = RD.iloc[7:,1:37]
    Q_Ex = RD.iloc[7:,0]  #real influent flow rate
    
    # slice by experiment
    RD_Ex1 = RD.iloc[7:,1:5]
    RD_Ex2 = RD.iloc[7:,5:9]
    RD_Ex3 = RD.iloc[7:,9:13]
    RD_Ex4 = RD.iloc[7:,13:17]
    RD_Ex5 = RD.iloc[7:,17:21]
    RD_Ex6 = RD.iloc[7:,21:25]
    RD_Ex7 = RD.iloc[7:,25:29]
    RD_Ex8 = RD.iloc[7:,29:33]
    RD_Ex9 = RD.iloc[7:,33:37]
    
    #concatenate to make the output data
    X_data = np.concatenate([RD_Ex1,RD_Ex2,RD_Ex3,RD_Ex4,RD_Ex5,RD_Ex6,RD_Ex7,RD_Ex8,RD_Ex9],axis=0)
    Q_data = np.concatenate([Q_Ex, Q_Ex, Q_Ex, Q_Ex, Q_Ex, Q_Ex, Q_Ex, Q_Ex, Q_Ex],axis=0)
    
    # RD_Ex.shape
    # data_x = np.reshape(np.ravel(RD_Ex), (1593, 4))
    
    # Q_Exp = np.savetxt("C:/Users/mark_/userdata/Output/Q_influent_exp.csv", Q_Ex, delimiter=',')

#### Create very long (6 months) L9 predictor variable training data array. Use to train NNs
# def NN_L9_traindata():
#     #RD = pd.read_excel('C:\\Users\\mark_\\Anaconda3\Data\SATW_project.xlsx', sheet_name='Database', sep=";", decimal=',')
#     RD = pd.read_excel('C:\\Users\\mark_\\Anaconda3\Data\SATW_project.xlsx', sheet_name='Database_L9', sep=";", decimal=',')
#     RD_Ex = RD.iloc[7:,0:37]
#     RD_Ex = pd.DataFrame(data = RD_Ex)#

#     Q_Ex = RD_Ex.iloc[7:,0]  # Real influent flow rate data

# # slice by experiment
#     RD_Ex1 = RD_Ex.iloc[7:,2:5] #2:5
#     RD_Ex2 = RD_Ex.iloc[7:,6:9]#6:9
#     RD_Ex3 = RD_Ex.iloc[7:,10:13]#10:13
#     RD_Ex4 = RD_Ex.iloc[7:,14:17]#14:17
#     RD_Ex5 = RD_Ex.iloc[7:,18:21]#18:21
#     RD_Ex6 = RD_Ex.iloc[7:,22:25]#22:25
#     RD_Ex7 = RD_Ex.iloc[7:,26:29]#26:29
#     RD_Ex8 = RD_Ex.iloc[7:,30:33]#30:33
#     RD_Ex9 = RD_Ex.iloc[7:,34:37]#34:37

# #concatenate to make the output data
#     M_data = np.concatenate([RD_Ex1,RD_Ex2,RD_Ex3,RD_Ex4,RD_Ex5,RD_Ex6,RD_Ex7,RD_Ex8,RD_Ex9],axis=0)#
#     M_data = pd.DataFrame(data = M_data)
#     Q_data = np.concatenate([Q_Ex, Q_Ex, Q_Ex, Q_Ex, Q_Ex, Q_Ex, Q_Ex, Q_Ex, Q_Ex],axis=0)
#     Q_data = pd.DataFrame(data = Q_data)
#     y_data = np.concatenate([RD_Ex.iloc[7:,1],RD_Ex.iloc[7:,5],RD_Ex.iloc[7:,9],RD_Ex.iloc[7:,13],RD_Ex.iloc[7:,17],RD_Ex.iloc[7:,21],RD_Ex.iloc[7:,25],RD_Ex.iloc[7:,29],RD_Ex.iloc[7:,33]], axis=0)
#     X_data = pd.concat([M_data, Q_data], axis=1, join='inner')
    
#     print('Mechanical set points', M_data.shape, 'Predictor data', X_data.shape, 'Influent', Q_data.shape,'Biogas', y_data.shape) #

    return #Qr #X_data, y_data 
    

    
def NN_QX_mockdata():
    RD = pd.read_excel('C:\\Users\\mark_\\mark_data\Input\SATW_project.xlsx', sheet_name='Mock experiment_E9')
    # RD = pd.read_excel('/scratch/mmccormi1/SATW_project.xlsx', sheet_name='Mock experiment_E9', index_col=None, engine="openpyxl") #sep=";",,  decimal=','Experiment_L4_V2
    # RD = np.genfromtxt(open("/scratch/mmccormi1/SATW_project.csv"), delimiter=';')
    RD_Ex = pd.DataFrame(data = RD) # RD_Ex = reference data from the experiment with additional data derived from T and Q (170 data points) plus mock data created using equation X
    ##print(RD_Ex.iloc[:170,[7]])
    IFR_ref_all = RD_Ex.iloc[:170,[7]].to_numpy() # Influent flow rate "experiment" in l/day.
    
    #df1, df2 = df1.align(df2)
    #RD_Ex.iloc[:170,[10]], RD_Ex.iloc[:170,[7]] = RD_Ex.iloc[:170,[10]].align(RD_Ex.iloc[:170,[7]])
    #CV_ref_kjl = RD_Ex.iloc[:170,[10]].to_numpy()   # CV reduction "experiment" in kJ/l.
    CV_ref_kjl = RD_Ex.iloc[:170,[10]].to_numpy()   # CV reduction "experiment" in kJ/l.
    print(CV_ref_kjl)
    CV_ref_kjd = np.multiply(RD_Ex.iloc[:170,[10]].to_numpy(), RD_Ex.iloc[:170,[7]].to_numpy())  #  kjd = kJ/d. ref -> CV reduction derived from the "Experiment" (kJ/l x l/day = kJ/day). RD_Ex.iloc[7:226,38]
    #CV_ref_kjd = RD_Ex.iloc[:170,[10]]*RD_Ex.iloc[:170,[7]]     #RD_Ex.iloc[:170,[10]].multiply(RD_Ex.iloc[:170,[7]], axis=0)  #  kjd = kJ/d. ref -> CV reduction derived from the "Experiment" (kJ/l x l/day = kJ/day). RD_Ex.iloc[7:226,38]
    print(CV_ref_kjd)
    # np.savetxt("C:/Users/mark_/userdata/Output/CV_ref_all.csv", CV_ref_all, delimiter=',')
    # CV_ref_all.plot()
    # plt.plot(CV_ref_all)
    Qinf = np.concatenate([IFR_ref_all, IFR_ref_all, IFR_ref_all, IFR_ref_all, IFR_ref_all, IFR_ref_all, IFR_ref_all, IFR_ref_all, IFR_ref_all], axis=0)#  Influent flow rate, experiment. Also same as X_data.iloc[:,3]   
    
    ##  Create the X_data arrays (Not MinMax scaled) for use in model development. 9 mock experiments x 170 days/experiement = 1530 rows
    # P = np.array([4, 12, 36, 1, 2.7, 11, 0.5, 1.8, 3.6])   #predictors 


    ##########################
    ##  Not MinMax scaled. P = predictors.  The order of the predictor lists is taken from the Taguchi L9 plan. 
##    P_ESD = np.array([4, 12, 36, 4, 12, 36, 4, 12, 36])#P_mm[:3].reshape((-1, 3), order='C') 
##    P_MAT = np.array([1, 2.7, 11, 2.7, 11, 1, 11, 1, 2.7])
##    P_HTD = np.array([0.5, 1.8, 3.6, 3.6, 0.5, 1.8, 1.8, 3.6, 0.5] )
##    
##    M_data_ESD = []
##    M_data_MAT = []
##    M_data_HTD = []
##    
##    for i in range(len(P_ESD)):     # # len = 9. Create the vector of ESD values for 9 mock experiments
##        for j in range(len(IFR_ref_all)):  # len = 170
##            M_data_ESD.append(P_ESD[i]) # len = 9 x 170 = 1530
##    
##    for i in range(len(P_MAT)):
##        for j in range(len(IFR_ref_all)):
##            M_data_MAT.append(P_MAT[i])
##    
##    for i in range(len(P_HTD)):
##        for j in range(len(IFR_ref_all)):
##            M_data_HTD.append(P_HTD[i])
##                
##    M_data = np.stack((M_data_ESD, M_data_MAT, M_data_HTD), axis = 1)
##    
##    X_data = np.c_[M_data, Qinf]  #Wow!, this works great.
##
##    label= ['ESD', 'MAT', 'HTD', ' Q influent']
##
##    fig, ax = plt.subplots()
##    ax.plot(X_data)
##    ax.set(xlabel='Samples from 9 concatenated mock experiments(Days)', ylabel='Value of mechanical parameter or of the influent flow rate',
##               title='Predictors used during 9 concatenated mock experiments \n 170 days per experiment x 9 mock experiments = 1530 days')
##    plt.legend(label)
##    #plt.savefig('C:/Users/mark_/Documents/02_Professional/03_Unil/SATW/MDPI_paper/complementary_information/figures/Fig_predictors_mock_exper.svg', format="svg")
##    plt.show()
##    # Load data above for use in all of the mock data creation methods
    
    
    ##  Create the MinMax scaled X_data
   
    #P_mm = np.select([P==4, P==12, P==36, P==1, P==2.7, P==11, P==0.5, P==1.8, P==3.6], [0, 0.25, 1, 0, 0.17, 1, 0, 0.42, 1])
##    P_mm_ESD = np.array([0, 0.25, 1, 0, 0.25, 1, 0, 0.25, 1])#P_mm[:3].reshape((-1, 3), order='C') 
##    P_mm_MAT = np.array([0, 0.17, 1, 0.17, 1, 0, 1, 0, 0.17])
##    P_mm_HTD = np.array([0, 0.42, 1, 1, 0, 0.42, 0.42, 1, 0] )
##      
##    M_data_mm_ESD = []
##    M_data_mm_MAT = []
##    M_data_mm_HTD = []
##    
##    for i in range(len(P_mm_ESD)):
##        for j in range(len(IFR_ref_all)):
##            M_data_mm_ESD.append(P_mm_ESD[i])
##    
##    for i in range(len(P_mm_MAT)):
##        for j in range(len(IFR_ref_all)):
##            M_data_mm_MAT.append(P_mm_MAT[i])
##    
##    for i in range(len(P_mm_HTD)):
##        for j in range(len(IFR_ref_all)):
##            M_data_mm_HTD.append(P_mm_HTD[i])
##                 
##    M_data_mm = np.stack((M_data_mm_ESD, M_data_mm_MAT, M_data_mm_HTD), axis = 1)
##    # M_data_mm_df = pd.DataFrame(M_data_mm)
##    
##    Qinf_mm = (Qinf [:] - Qinf [:].min()) / (Qinf [:].max() - Qinf [:].min())
##    #Qinf_mm = Qinf_mm.to_numpy()
##    
##    X_data_mm = np.c_[M_data_mm, Qinf_mm]  #Wow!, this works great.

    #######################
    ###  Fit a 4th degree polynomial to reference experimental data using Scipy, Optimize curve_fit library
    X_mech = np.array(170*[(12, 2.7, 1.8)])  # These are the true values used during on-site data collection (experiments) 
    X_ref = np.c_[X_mech,IFR_ref_all]  #Wow!, this works great.
    y_ref = CV_ref_kjd
    
    # # Fit objective function
    x = X_ref#.flatten()
    print('x = X_ref shape',x.shape)
    y= np.asarray(y_ref).flatten()
##    ye = y + np.random.normal(size=len(x), scale=0.1)  # add some error to y
##    print(y.shape)
##
##
##    # a : spherical diameter
##    # b : material type
##    # c : height to diameter ratio
##    # d : influent flow rate
##    # e : y intercept
##    
##    def objective(x, a, b, c, d, e):
##        #return a*x[:,0] + b*x[:,1] + c*x[:,2] + d*x[:,3] +e
##        return a*x[0] + b*x[1] + c*x[2] + d*x[3] +e
##    # # curve fit
##    yerr = np.array(170*[(3)])  # error in the predictions
##    popt, pcov  = curve_fit(objective, x.T, ye, sigma=yerr)  #_  ye or y, check it.
##    # # summarize the parameter values
##    a, b, c, d, e = popt
##    print('y = %.2f * x + %.2f * x + %.2f * x + %.2f * x + %.2f' % (a, b, c, d, e))
##    perr = np.sqrt(np.diag(pcov)) # To compute one standard deviation errors on the parameters.
##    print(perr)#'one standard deviation errors on the parameters + %.2f' % (pcov))
##    
##    polyref_a = objective(x.T, a, b, c, d, e )
##    print(polyref_a.shape)
##
##
##    rms_poly = sqrt(mean_squared_error(y, polyref_a.reshape(-1, 1)))
##    #print("Scipy, Polynomial fit to experimental data rms error is: " + str(rms_poly))
##    print("Scipy, Polynomial fit to experimental data rms error is: %.2f " % (rms_poly))
##    
##    reg_poly = LinearRegression().fit(y.reshape(-1, 1), polyref_a.reshape(-1, 1)) #, fit_intercept=True
##    #print("Scipy, Polynomial fit to experimental data R2 is: " + str(reg_poly.score(y_ref, polyref_a)))   
##    print("Scipy, Polynomial fit to experimental data R2 is: %.2f"  % (reg_poly.score(y_ref, polyref_a)))
##        
##    fig, ax = plt.subplots()
##    ax.scatter(y, polyref_a)
##    ax.set(xlabel='Reference CV reduction (kJ/day)', ylabel='Predicted CV reduction (kJ/day)',
##               title='Calorific value reduction, 4th degree polynomial fit to reference \n "experimental" data using Scipy, Optimize, curve_fit library')
##    ax.text(20, 190,'y = %.2f * x + %.2f * x + %.2f * x + %.2f * x + %.2f' % (a, b, c, d, e))
##    ax.text(20, 170,"Polynomial fit to experimental data rms error is: %.2f " % (rms_poly))
##    ax.text(20, 160,"Polynomial fit to experimental data R2 is: %.2f"  % (reg_poly.score(y_ref, polyref_a)))
##    plt.xlim((0,200))
##    plt.ylim((0,200))
##    plt.show()

##    fit_x = np.arange(min(polyref_a), max(polyref_a),1)
##    
##    plt.plot(polyref_a)
##    plt.plot(objective(fit_x.flatten(), a, b, c, d, e ))

##
##    ###############
##    ###########  Use the MatPlotliblinear regression package
##    mdlr_ref = LinearRegression(fit_intercept=True).fit(X_ref,y_ref)
##    Wr_ref = mdlr_ref.coef_
##    Ir_ref = mdlr_ref.intercept_
##    polyref = []
##    
##    for k in range(len(y_ref)):
##        #plt.plot(Wr_ref[0]*X_ref[0]+Wr_ref[1]*X_ref[1]+Wr_ref[2]*X_ref.T[2]+Wr_ref[3]*X_ref[3]+Ir_ref)
##        #plt.plot(Wr_ref[0]*y_ref[k]+Wr_ref[1]*y_ref[k]+Wr_ref[2]*y_ref[k]+Wr_ref[3]*y_ref[k]+Ir_ref)
##        polyref.append(Wr_ref[0][0]*IFR_ref_all[k]+Wr_ref[0][1]*IFR_ref_all[k]+Wr_ref[0][2]*IFR_ref_all[k]+Wr_ref[0][3]*IFR_ref_all[k]+Ir_ref)
##        #plt.scatter((1.077*IFR_ref_all[k]+0.2424*IFR_ref_all[k]+0.1616*IFR_ref_all[k]+0.2124*IFR_ref_all[k]+Ir_ref), marker='+', linestyle='')
##
##    polyref_mpl = np.concatenate([np.array(j) for j in polyref], axis = 0)#[i for polyref(i) in len(polyref)]
##    
##    fig, ax = plt.subplots()
##    labels = ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'Ref']
##    ax.scatter(y_ref, polyref_mpl)
##    #ax.plot(np.asarray(PME_CVred_res)[:,j], marker='.', linestyle='')  #None
##    ax.set(xlabel='Reference CV reduction (kJ/day)', ylabel='Predicted CV reduction (kJ/day)',
##               title='Calorific value reduction, 4th degree polynomial fit to reference \n "experimental" data using MatPlotlib LinearRegression library')
##    #ax.grid()
##    # # ax.plot(CV_ref_all, marker='+', linestyle='')
##    plt.xlim((0,200))
##    plt.ylim((0,200))
##    # plt.legend(labels)
##    # fig.savefig("test.png")
##    plt.show()
##    
##    rms_poly = sqrt(mean_squared_error(y_ref, polyref_mpl))   # mpl = MatPlotlib
##    #print("MatPlotlib, Polynomial fit to experimental data rms error is: " + str(rms_poly))
##    print("MatPlotlib, Polynomial fit to experimental data rms error is: %.2f" %(rms_poly))
##    
##    reg_poly = LinearRegression().fit(y_ref, polyref_mpl) #, fit_intercept=True
##    #print("MatPlotlib, Polynomial fit to experimental data R2 is:" + str(reg_poly.score(y_ref, polyref_mpl)))   
##    print("MatPlotlib, Polynomial fit to experimental data R2 is: %.2f " %(reg_poly.score(y_ref, polyref_mpl)))  

    ##########################
    # ##### Generate mock experimental data from mock experiments. Mock experimental data is enhanced surrogate data. Surrogate data was derived from acquired data.  2 different methods. 
    # #######
    
    # ### Method N°1: Use the Scipy curve_fit package to fit 4th degree polynomial to the surrogate data. This function predicts CV reduction from 4 predictors (ESD, MAT, HDR, Qin).
    # ### Change the polynomial coefficients (predictors) one at a time to generate responses for each of the 9 experiments. 
    
    #  # exper = [(ESD==4, MAT==1.0, HDR==0.5), (ESD==12, MAT==2.7, HDR==1.8),(ESD==36, MAT==12.0, HDR==4.0),(ESD==4, MAT==2.7, HDR==4),(ESD==12.0, MAT==12.0, HDR==0.5),
    # #                (ESD==36, MAT==1.0, HDR==1.8),(ESD==4, MAT==12.0, HDR==1.8),(ESD==12, MAT==1.0, HDR==4.0),(ESD==36, MAT==2.7, HDR==0.5)]
    # # IFR_PME = []
    
    # from satw_mlp_v5 import  AFBR_MLP_model
    # ## Surrogate predictor dataset. Change one predictor at a time
    # M1 = np.c_[[np.array([4, 2.7, 1.8]) for j in range(len(IFR_ref_all))], IFR_ref_all]
    # M2 = np.c_[[np.array([36, 2.7, 1.8]) for j in range(len(IFR_ref_all))], IFR_ref_all]
    # M3 = np.c_[[np.array([12, 1, 1.8]) for j in range(len(IFR_ref_all))], IFR_ref_all]
    # M4 = np.c_[[np.array([12, 12, 1.8]) for j in range(len(IFR_ref_all))], IFR_ref_all]
    # M5 = np.c_[[np.array([12, 2.7, 0.5]) for j in range(len(IFR_ref_all))], IFR_ref_all]
    # M6 = np.c_[[np.array([12, 2.7, 4.0]) for j in range(len(IFR_ref_all))], IFR_ref_all]
    
    # X_data = np.r_[M1, M2, M3, M4, M5, M6]
    # Xscaler = MinMaxScaler()
    # Xscaler.fit(X_data)
    # X_data_mm = Xscaler.transform(X_data)
    
    # ## Surrogate response dataset.
    # y_data = np.concatenate([CV_ref_all, CV_ref_all, CV_ref_all, CV_ref_all, CV_ref_all, CV_ref_all], axis=0)
    # yscaler = MinMaxScaler()
    # yscaler.fit(y_data)
    # y_data_mm = yscaler.transform(y_data)
    
    
    # ## Random seed to set random state for each MLP build
    # rng = default_rng()
    # seed_50X = rng.integers(low=1, high=99, size=50)
    # rmse_5_10 = []
    
    
    # for s in range(50):
    #     X_train_mm, X_test_mm, y_train_mm, y_test_mm = train_test_split(X_data_mm, y_data_mm, test_size=0.3, random_state= seed_50X[s], shuffle = True, stratify= None)
    #     Yhat, Yhat_v, test_rmse = AFBR_MLP_model(X_train_mm, y_train_mm, X_test_mm, y_test_mm)
    #     rmse_5_10.append(test_rmse)
        
    # X_axis = np.arange(1,66,1)
    # fig, ax = plt.subplots()
    # ax.set(xlabel='Replicate run', ylabel='RMSE', title='RMSE versus replicate MLP build')      
    # ax.scatter(X_axis, rmse_5_10)
    # # fig.savefig("test.png")
    # plt.show()
    
    # # plt.scatter(X_axis, rmse_5_10)

    ################
    ### Method N°2: Use the 4th degree polynomial that was previously generated from surrogate data. This function predicts CV reduction from 4 predictors (ESD, MAT, HDR, Qin).
    ### Change the polynomial coefficients (predictors) one at a time to generate responses for each of the 9 experiments. 
    
     # exper = [(ESD==4, MAT==1.0, HDR==0.5), (ESD==12, MAT==2.7, HDR==1.8),(ESD==36, MAT==12.0, HDR==4.0),(ESD==4, MAT==2.7, HDR==4),(ESD==12.0, MAT==12.0, HDR==0.5),
    #                (ESD==36, MAT==1.0, HDR==1.8),(ESD==4, MAT==12.0, HDR==1.8),(ESD==12, MAT==1.0, HDR==4.0),(ESD==36, MAT==2.7, HDR==0.5)]
    # IFR_PME = []
    #     Values from fit to reference experiment data.     Means from LOO replications (100)
##    PME_CVred = []
##    PME_CVred_res = []
##    reps=10
##    for n in range(reps):  # set the number of replications
##        def objective(x, a, b, c, d, e):
##                #return a*x[:,0] + b*x[:,1] + c*x[:,2] + d*x[:,3] +e
##            return a*x[0] + b*x[1] + c*x[2] + d*x[3] +e
##            # # curve fit
##        ye = y + np.random.normal(size=len(x), scale=0.1)  # add some error to y
##        yerr = np.array(170*[(10)])  # error in the predictions
##        popt, pcov  = curve_fit(objective, x.T, ye, sigma=yerr)  #_  ye or y, check it.
##            # # summarize the parameter values
##        a, b, c, d, e = popt
##
##        
##        Wr0 = a#    spherical diameter                                         #0.372682
##        Wr1 = b#    material type                                           #0.132514
##        Wr2 = c#    height to diameter ratio                                          #0.00210878
##        Wr3 = d#    influent flow rate                                          #-0.0603423
##        Ir  = e#    y intercept                                              #0.4857991408
##
##        L1=0.999 # A coefficient to reduce the value of the predictor 
##        L2=1     #A different coefficient to reduce the value of the predictor 
##        H=1.001      #A coefficient to increase the value of the predictor 
##
##        E_ref   =  [Wr0, Wr1, Wr2, Wr3]   # Base case. Coeficient from the polynomial regression of MinMax scaled data.
##        E_1 =  [L1*Wr0, Wr1, Wr2, Wr3]     # Low
##        E_2 =  [H*Wr0, Wr1, Wr2, Wr3]      # High
##        E_3 =  [Wr0, L1*Wr1, Wr2, Wr3]     # Low
##        E_4 =  [Wr0, H*Wr1, Wr2, Wr3]      # High
##        E_5 =  [Wr0, Wr1, L1*Wr2, Wr3]     # Low
##        E_6 =  [Wr0, Wr1, H*Wr2, Wr3]      # High
##        #Wr_7 =  [L2*Wr0, Wr1, Wr2, Wr3]     # Low
##        #Wr_8 =  [Wr0, L2*Wr1, Wr2, Wr3]     # Low
##        #Wr_9 =  [Wr0, Wr1, L2*Wr2, Wr3]     # Low
##        
##        # # Wr   =  [0.372682, 0.132514, 0.00210878, -0.0603423]   # Base case. Coeficient from the polynomial regression of MinMax scaled data.
##        # # Wr_1 =  [L1*0.372682, 0.132514, 0.00210878, -0.0603423]     # Low
##        # # Wr_2 =  [H*0.372682, 0.132514, 0.00210878, -0.0603423]      # High
##        # # Wr_3 =  [0.372682, L1*0.132514, 0.00210878, -0.0603423]     # Low
##        # # Wr_4 =  [0.372682, H*0.132514, 0.00210878, -0.0603423]      # High
##        # # Wr_5 =  [0.372682, 0.132514, L1*0.00210878, -0.0603423]     # Low
##        # # Wr_6 =  [0.372682, 0.132514, H*0.00210878, -0.0603423]      # High
##        # # Wr_7 =  [L2*0.372682, 0.132514, 0.00210878, -0.0603423]     # Low
##        # # Wr_8 =  [0.372682, L2*0.132514, 0.00210878, -0.0603423]     # Low
##        # # Wr_9 =  [0.372682, 0.132514, L2*0.00210878, -0.0603423]     # Low
##        
##        
##        exper2 = [E_1, E_2, E_3, E_4, E_5, E_6,  E_ref]   #The coeficients of the mock experiments and the reference experiment.Wr_7, Wr_8, Wr_9,
##        
##        # X_test = [(4, 1.0, 0.5), (12, 2.7, 1.8),(36, 12.0, 4.0),(4, 2.7, 4),(12.0, 12.0, 0.5),
##        #                (36, 1.0, 1.8),(4, 12.0, 1.8),(12, 1.0, 4.0),(36, 2.7, 0.5)]
##        
##
##        ## Generate CV reduction data from 9 mock experiments. Modify the reference predictors for each experiement.
##        for h in range(len(exper2)):  #experiments
##            for k in range(len(X_ref)):#len(X_test_mm)
##                # IFR_PME.append(X_test_mm[k,0]*Wr[0] + X_test_mm[k,1]*Wr[1] + X_test_mm[k,2]*Wr[2] + X_test_mm[k,3]*Wr[3]+Ir)
##                PME_CVred.append(X_ref[k,0]*exper2[h][0] + X_ref[k,1]*exper2[h][1] + X_ref[k,2]*exper2[h][2] + X_ref[k,3]*exper2[h][3]+Ir)
##        
##        # PME_CVred_res = np.split(np.asarray(PME_CVred), len(exper1), axis=0)
##    plt.figure()
##    plt.plot(PME_CVred)  #  Vector: 170 x (N° of mock experiments + ref exp) x n° of replicates. Eg 170 x 7 x 10 = 11900
##    plt.show()
##    print(np.asarray(PME_CVred).shape)
####    PME_CVred_mock = np.asarray(PME_CVred).reshape(-1, len(exper2), order = 'F').T
####    print(PME_CVred_mock.shape)
####    plt.figure()
####    plt.plot(PME_CVred_mock)
####    plt.show()
##    #PME_CVred_res = np.asarray(PME_CVred).reshape(np.multiply(len(X_ref),reps),-1, order = 'F')
##    PME_CVred_res = np.asarray(PME_CVred).reshape(-1, 1190 , order = 'F') #len(X_ref)
##    print(PME_CVred_res.shape)
##    plt.figure()
##    plt.plot(PME_CVred_res)
##    plt.show()
##    
##    q, r, s, t, u, v, w = np.hsplit(PME_CVred_res, 7)
##    print(np.asarray(q).shape)
##    plt.figure()
##    plt.plot(q)
##    plt.show()
##
##    PME_CVred_mock = np.stack([q, r, s, t, u, v, w], axis=1)
##    print(np.asarray(PME_CVred_mock).shape)
##    plt.figure()
##    plt.plot(PME_CVred_mock)
##    plt.show()
##    
    
##    fig, ax = plt.subplots()
##    labels = ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'Ref']  # 'M7', 'M8', 'M9',
##    r = 0
##    s=0
##    for j in range(len(exper2)):
##        for r in range(reps):
##            ax.plot(np.asarray(PME_CVred_res)[r,:], marker='.', linestyle='')  #None
####    for r in range(reps):
####        for j in range(len(exper2)):
####            ax.plot(np.asarray(PME_CVred_res)[r,s:(s+170)], marker='.', linestyle='')  #None
##            ax.set(xlabel='Time(days)', ylabel='MinMax scaled CV reduction (kJ/day)',
##                title='Calorific value reduction, 6 or 9 mock experiments \n using polynomial coefficients changed one at a time')
##            s+=170
##            #ax.grid()
##    # ax.plot(CV_ref_all, marker='+', linestyle='')
##    # plt.ylim((0,1))
##    plt.legend(labels)
##    # fig.savefig("test.png")
##    plt.show()
##
##    fig, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(12,12), sharey=True, tight_layout=True, gridspec_kw={'width_ratios': [4, 1]})
##    
##    labels = ['M1', 'M2', 'M3', 'M4', 'M5', 'M6',  'Ref']  #'M7', 'M8', 'M9',
##    for j in range(len(exper2)):
##        ax1.plot(np.asarray(PME_CVred_res)[:,j], marker='.', linestyle='')  #None
##        
##        ax1.set(xlabel='Time(days)', ylabel='CV reduction (kJ/day)',
##               title='Calorific value reduction, 6 or 9 mock experiments \n using polynomial coefficients changed one at a time')
##        ax1.grid()
##    #ax1.plot(CV_ref_all, marker='x', linestyle='', markersize=10, c='k') #  Add _mm if required.
##    
##    n_bins=17
##    for k in range(len(exper2)):
##        ax2.hist(np.asarray(PME_CVred_res)[:,k], bins= n_bins, orientation='horizontal', stacked=True)
##        # axs[1].hist(IFR_M1[:,1], bins= n_bins)
##        ax2.set(xlabel='Bin count') #ylabel='CV reduction (kJ/day)',
##                # title='Calorific value reduction, 9 mock experiments \n Using coefficients based on litterature')
##        # ax2.grid()
##    # ax1.plot(CV_ref_all, marker='+', linestyle='') #  Add _mm if required. 
##    plt.legend(labels)
##    # fig.savefig("test.png")
##    # plt.savefig('C:/Users/mark_/Documents/02_Professional/03_Unil/SATW/MDPI_paper/MDPI_article/MDPI_article_v23/figures/F2_cv_reduction_l9_b.svg', format="svg")
##    plt.show()
##    
##    # #plt.plot(np.asarray(IFR_PME_res).T)
##    # # IFR_M1 = np.concatenate([np.asarray(S), np.asarray(M), np.asarray(H)], axis = 1)
##    
##    # # IFR_M1 = (np.asarray(S) + np.asarray(M) + np.asarray(H)).reshape(-1,len(exper), order = 'F')
##   
##    
    #######################################
    ###   Method N°3: Use the method inspired from the litterature (Based on previous Excel spreadsheet method)
    #####
    S=[]    #  ESD component of total CV reduction
    M=[]    #  MAT component of total CV reduction
    H=[]    # HDR component of total CV reduction
   
    Qmax = np.max(IFR_ref_all) #516 l/day # Maximum infuent flow rate
    print('Qmax', Qmax)
    
    i=0
    h=0
    ## Settings that make a good correlation between CV_ref_kjd, IFR_M1[:,1]: SF = 4. SC=MF=HF=20. 0.33*IFR_1[i]/Qmax when 12, 2.7 or 1.8.
    SF = 1550# Scale factor to adjust the value predicted by each of the 3 features.  

    SC = 1# sphere diameter adjustment factor
    SC1 = 820#1.1#4     # Scale factor to 
    SC2 = -770#1.1#1.5  # Scale factor to increase CVred in non-linear proportion to flow rate when void space is large
    SC3 = 350#0.9#2.5 # Threshold below which the CVred decreases in negative exponential proportion to flow rate when Q is below SC3*Qmax
    SC4 = 0.4# 0,4*516 = 204. Threshold below which the CVred increases exponentially in proportion to flow rate when Q is below SC4*Qmax and SD = 4 because small SD is well adapted to low flow rates.
    SC12 = 1  # Scale factor on the reference experiment results to calibrate with other response variable effects.
    
    MF = 1#0.185   #  Material type adjustment factor
    FR = 1#2    # reduced flow effect factor
    IF = 1.2  # IF = Is Foam ajustment factor
    IFC = -200 #-700 #IFC = Is Foam ajustment factor to account for clogging when flow rate is less than SC4*Qmax
    PVC = 230#1  # PVC = is PVC ajustment factor
    MF27 = 1 # Scale factor on the reference experiment results to calibrate with other response variable effects.

    HF = 1#0.2  # HDR selector ajustment factor
    HDR1 = 170#  # HDR ajustment factor on 0.5 when flow rate is low and well adapted to height
    HDR2 = 0.2#1.0  #2 HDR ajustment factor 
    HDR3 = 170#1.0#2.8  # HDR ajustment factor on 3.6 when flow rate is high and well adapted to height
    HDR18 = 1 # Scale factor on the reference experiment results to calibrate with other response variable effects.

    #  Predictors. 9 experiments with values from the range described in the MDPI article text.
    exper_lit = [(4.0, 1.0, 0.5), (12.0, 2.7, 1.8),(36.0, 11.0, 3.6),(4.0, 2.7, 3.6),(12.0, 11.0, 0.5),
                    (36.0, 1.0, 1.8),(4.0, 11.0, 1.8),(12.0, 1.0, 3.6),(36.0, 2.7, 0.5)]

    A = 170*[(4.0, 1.0, 0.5)]
    B = 170*[(12.0, 2.7, 1.8)]
    C = 170*[(36.0, 11.0, 3.6)]
    D = 170*[(4.0, 2.7, 3.6)]
    E = 170*[(12.0, 11.0, 0.5)]
    F = 170*[(36.0, 1.0, 1.8)]
    G = 170*[(4.0, 11.0, 1.8)]
    Ha = 170*[(12.0, 1.0, 3.6)]
    I = 170*[(36.0, 2.7, 0.5)]



    M_data = np.concatenate((A,B,C,D,E,F,G,Ha,I), axis=0)
    #print('A',A)
##    A = []           
##    M_data = [] #exper_lit
####    np.concatenate([IFR_ref_all,
##    for e in range(0,len(exper_lit)):
##        new = 170*[exper_lit[e]]
##        #print('new', new)
##        M_data = np.concatenate((M_data, np.asarray(new[:,:3])), axis=0) 
##        #M_data.append( for e in exper_lit))
    print('M_data as array', np.asarray(M_data))
    print('M_data length', np.asarray(M_data).shape)
                    
    X_data = np.c_[M_data, Qinf]  #Wow!, this works great.
    print('X_data', X_data)

    label= ['ESD', 'MAT', 'HTD', ' Q influent']
    fig, ax = plt.subplots()
    ax.plot(X_data)
    ax.set(xlabel='Samples from 9 concatenated mock experiments(Days)', ylabel='Value of mechanical parameter or of the influent flow rate',
               title='Predictors used during 9 mock experiments in series\n 170 days per experiment x 9 mock experiments = 1530 days')
    plt.legend(label)
    plt.savefig('C:/Users/mark_/Documents/02_Professional/03_Unil/SATW/MDPI_paper/complementary_information/figures/Fig_predictors_mock_exper.svg', format="svg")
    plt.show()
    
##    mu_1, sigma_1 = 60, 6
##    mu_2, sigma_2 = 120, 12
##    mu_3, sigma_3 = 180, 18
    np.random.seed(42)
    mu, sigma = 1, 0.01#0.075 Introduce randomness in the influent flow rates between separate experiments. mean and standard deviation= 170, 10
    IFR_1 =IFR_2 = IFR_3 =IFR_ref_all[:, 0]* np.random.normal(mu, sigma, 170) #np.multiply(IFR_ref_all, np.random.normal(mu, sigma, 170))#,np.random.normal(mu, sigma, 35),np.random.normal(mu, sigma, 35)), axis=0)
##    IFR_2 = IFR_ref_all[:, 0]#*np.random.normal(mu, sigma, 170)
##    IFR_3 = IFR_ref_all[:, 0]#*np.random.normal(mu, sigma, 170)

    for h in range(0,len(exper_lit)):  #Mock experiments kjl x l/day = kJ/day
        for i in range(len(IFR_1)):
            S.append(np.multiply((CV_ref_kjl[i]),(                  # aprox 0 - 0,5 kjl
            1+
            #((SC1*exper_lit[h][0]*np.sin(3.14*IFR_1[i]/Qmax)) +
            #np.multiply(exper_lit[h][0],IFR_1[i]/Qmax) +
            np.where(exper_lit[h][0]==12.0, SF*0.33*IFR_1[i]/Qmax, 0)+ # 127/516 = 0,25. 0.25x0,25 = 0,06. 0.06x333/3 = 20 kJ/day (min).  0,5x1x320 = 160 kJ/day (max)
            np.where(exper_lit[h][0]==36.0 and IFR_1[i] > 0.8*Qmax , SF*0.1, 0)+
            np.where(exper_lit[h][0]==36.0 and IFR_1[i] > SC4*Qmax , 0.8*SF*0.33*(1-(-IFR_1[i]/Qmax)), 0)+  #np.sin((3.14/2)*(IFR_1[i]/Qmax))
            np.where(exper_lit[h][0]==36.0 and IFR_1[i] < SC4*Qmax , 0.1*SF*0.33*np.exp(-IFR_1[i]/Qmax), 0) +   # -2.5
            np.where(exper_lit[h][0]==36.0 and IFR_1[i] < 0.45*Qmax , -SF*0.05, 0) + 
####            #np.where(exper_lit[h][0]==12.0, SC3*np.sin(3.14*IFR_1[i]/Qmax), SC3*IFR_1[i]))*SC)*SF)
####            #np.where(exper_lit[h][0]==4.0 and IFR_1[i] < SC4*Qmax ,SC1*(np.exp(-IFR_1[i]/Qmax)), 0) +   #1.72
####            #np.where(exper_lit[h][0]==4.0, SC3*np.sin(3.14*IFR_1[i]/Qmax), SC3*IFR_1[i]))*SC)*SF)
            np.where(exper_lit[h][0]==4.0 and IFR_1[i] < SC4*Qmax , 0.4*SF*0.33*np.exp(-IFR_1[i]/Qmax), 0)+
            np.where(exper_lit[h][0]==4.0 and IFR_1[i] < 0.8*Qmax , 2*(SF*4/12)*0.33*IFR_1[i]/Qmax, 0)+
            np.where(exper_lit[h][0]==4.0 and IFR_1[i] > 0.8*Qmax , 0.01*SF*0.33*(1-np.exp(IFR_1[i]/Qmax)),0)+  #)*SC)*SF)
            np.where(exper_lit[h][0]==4.0 and IFR_1[i] < 0.45*Qmax , -SF*0.01, 0)+
            0)*SC)*1)
             
            M.append(np.multiply((CV_ref_kjl[i]),(
            1+
            #(np.multiply(exper_lit[h][1],
            #(np.exp(exper_lit[h][0]*np.true_divide(IFR_1[i],(Qmax*FR)))+ #FR* np.true_divide(IFR_1[i],Qmax) +
            #np.where(exper_lit[h][1]==11.0 ,IF*IFR_1[i], IFR_1[i])+
            #np.multiply((36/11)*exper_lit[h][1], IFR_1[i]/Qmax) +
            np.where(exper_lit[h][1]==2.7, SF*0.33*IFR_1[i]/Qmax, 0) + # MF27*np.exp(IFR_1[i]/Qmax)
            np.where(exper_lit[h][1]==11.0 and IFR_1[i]> 0.8*Qmax, SF*0.2, 0)+ 
            np.where(exper_lit[h][1]==11.0 and IFR_1[i]> 0.3*Qmax, 0.3*SF*0.33*np.sin((3.14/2)*(IFR_1[i]/Qmax)), 0) +
            np.where(exper_lit[h][1]==11.0 and IFR_1[i]< 0.3*Qmax, -0.1*SF*0.33*np.exp(-IFR_1[i]/Qmax), 0)+  #-1.7
            np.where(exper_lit[h][1]==11.0 and IFR_1[i]< 0.3*Qmax, -0.05*SF*0.33, 0)+ 
            np.where(exper_lit[h][1]==1.0 and IFR_1[i]< SC4*Qmax, 0.5*SF*0.33*np.exp(-IFR_1[i]/Qmax), 0)+
            np.where(exper_lit[h][1]==1.0 and IFR_1[i]< 0.8*Qmax, 11*(SF*1/11)*0.33*IFR_1[i]/Qmax, 0)+
            np.where(exper_lit[h][1]==1.0 and IFR_1[i]> 0.8*Qmax, 0.03*SF*0.33*(-np.exp(-IFR_1[i]/Qmax)),0)+    #)*MF)*SF)
            #np.where(exper_lit[h][1]==1.0 and IFR_1[i]> 0.8*Qmax, -SF*0.03, 0)+
            0)*SC)*1)

            H.append(np.multiply((CV_ref_kjl[i]), (
            #np.multiply((36/4)*exper_lit[h][2], IFR_1[i]/Qmax) +
            1+
            np.where(exper_lit[h][2]==1.8, SF*0.33*IFR_1[i]/Qmax, 0) + # HDR18*np.exp(IFR_1[i]/Qmax)
            np.where(exper_lit[h][2] == 0.5 and IFR_3[i]<Qmax*0.5/3.6, SF*0.33*IFR_3[i]/Qmax, 0)+  #0.2*(1-np.exp(-IFR_1[i]/50))
            np.where(exper_lit[h][2] == 0.5 and IFR_3[i]>0.5/3.6, 0.4*SF*0.33*IFR_3[i]/Qmax, 0)+
####            #np.where(exper_lit[h][2] == 1.8 and IFR_3[i]<Qmax*1.8/3.6, HDR2*IFR_3[i]/Qmax, 0)+   # 1.8/3.6 = 0.5 .  516*1.8/3.6 = 258 and IFR_3[i]>Qmax*0.5/3.6 
####            #np.where(exper_lit[h][2] == 1.8 and IFR_3[i]>Qmax*1.8/3.6, -HDR1*IFR_3[i]/Qmax, 0)+
####            #np.where(exper_lit[h][2] == 1.8 and IFR_3[i]>Qmax*1.8/3.6, HDR2*np.exp(-IFR_1[i]/100), 0)+
            np.where(exper_lit[h][2] == 3.6 and IFR_3[i]< Qmax*1.8/3.6, SF*0.33*IFR_3[i]/Qmax, 0)+
            np.where(exper_lit[h][2] == 3.6 and IFR_3[i]>Qmax*1.8/3.6, 0.8*SF*0.33*IFR_3[i]/Qmax, 0) +    #)*HF)*SF) #and IFR_3[i]>Qmax*0.5/3.6
            #np.where(exper_lit[h][2] == 3.6 and IFR_3[i]>Qmax*0.5/3.6 and IFR_3[i]>Qmax*1.8/3.6, HDR1*IFR_3[i]/Qmax, HDR3*np.exp(-IFR_1[i]/100)))*HF)*SF)
            0)*SC)*1)
            ####
##    for h in range(3,6):  #Mock experiments 4 - 6
##        for i in range(len(IFR_2)):
##            S.append(np.multiply(CV_ref_kjl[i], (np.sin(3.14*IFR_2[i]/Qmax)/SC1 + np.exp((exper_lit[h][0]*IFR_2[i]/Qmax)/SC2) + np.where(IFR_2[i]< 0.2*Qmax, SC3*IFR_2[i], IFR_2[i])))*SC*SF)
##            
##            M.append(np.multiply(CV_ref_kjl[i], (np.multiply(exper_lit[h][1], IFR_2[i]/Qmax) + np.multiply(NF*exper_lit[h][1], IFR_2[i], where=(exper_lit[h][1]!=11.0))+ np.where(IFR_2[i]> 0.8*Qmax, np.where(exper_lit[h][1] < 2.7, np.multiply(-0.25, IFR_2[i]), IFR_2[i]), IFR_2[i])))*MF*SF)  #
##            
##            H.append(np.multiply(CV_ref_kjl[i],((np.multiply(HDR1*IFR_3[i]/Qmax, exper_lit[h][2], where=(IFR_3[i]<Qmax*0.5/3.6)))+ np.multiply(HDR1*IFR_3[i]/Qmax, exper_lit[h][2], where=(IFR_3[i]>Qmax*0.5/3.6)))+ np.multiply(CV_ref_kjl[i], np.multiply(HDR1*IFR_3[i]/Qmax, exper_lit[h][2], where=(IFR_3[i]>Qmax*1.8/3.6))))*HF*SF)
##
##    for h in range(6,9):  #Mock experiments 7 - 9
##        for i in range(len(IFR_3)):
##            S.append(np.multiply(CV_ref_kjl[i], (np.sin(3.14*IFR_3[i]/Qmax)/SC1 + np.exp((exper_lit[h][0]*IFR_3[i]/Qmax)/SC2) + np.where(IFR_3[i]< 0.2*Qmax, SC3*IFR_3[i], IFR_3[i])))*SC*SF)
##
##            M.append(np.multiply(CV_ref_kjl[i], (np.multiply(exper_lit[h][1], IFR_3[i]/Qmax) + np.multiply(NF*exper_lit[h][1], IFR_3[i], where=(exper_lit[h][1]!=11.0))+ np.where(IFR_3[i]> 0.8*Qmax, np.where(exper_lit[h][1] < 2.7, np.multiply(-0.25, IFR_3[i]), IFR_3[i]), IFR_3[i])))*MF*SF)  #
##            
##            H.append(np.multiply(CV_ref_kjl[i], ((np.multiply(HDR1*IFR_3[i]/Qmax, exper_lit[h][2], where=(IFR_3[i]<Qmax*0.5/3.6)))+ np.multiply(HDR1*IFR_3[i]/Qmax, exper_lit[h][2], where=(IFR_3[i]>Qmax*0.5/3.6)))+ np.multiply(CV_ref_kjl[i], np.multiply(HDR1*IFR_3[i]/Qmax, exper_lit[h][2], where=(IFR_3[i]>Qmax*1.8/3.6))))*HF*SF)#+ np.where(IFR_ref_all[i]> 0.8*Qmax, IFR_ref_all[i], IFR_ref_all[i]/Qmax)))
##                                 
    # IFR_M1 = np.concatenate([np.asarray(S), np.asarray(M), np.asarray(H)], axis = 1)
    print('S',np.asarray(S))
    print('M',np.asarray(M))
    print('H',np.asarray(H))
    print('H shape:', np.asarray(H).shape)

    IFR_Mtest = np.sum((np.asarray(S) + np.asarray(M) + np.asarray(H)), axis=0)   # This is the mock y_data (predicted CV reduction) without the reference experiment data (However, E2(12, 2.7, 1.8) = same predictors as reference experiment)
##    print('IFR_Mtest shape (Sum of S,M,H)?:', IFR_Mtest.shape)
    print('S sum', np.sum(S))
    print('M sum', np.sum(M))
    print('H sum', np.sum(H))
    print('IFR_Mtest sum', np.sum(IFR_Mtest))
##    print(IFR_Mtest)
##    print(np.asarray(S).shape)
    #IFR_Mtest_v = np.vstack((np.asarray(S), np.asarray(M), np.asarray(H)))
    #print('IFR_Mtest vstack shape',IFR_Mtest_v.shape)
    IFR_Mtest = np.concatenate((np.asarray(S), np.asarray(M), np.asarray(H)), axis=1).T
    print('IFR_Mtest concat shape',IFR_Mtest.shape)
    IFR_Mtest = np.mean(IFR_Mtest, axis=0).reshape(-1,1)
    print('IFR_Mtest means shape',IFR_Mtest.shape)
    print('IFR_Mtest ', IFR_Mtest)

    plt.figure()
    plt.plot(IFR_Mtest)
    plt.title('CVred, all 9 mockexperiments (IFR_Mtest)')
##    plt.savefig('C:/Users/mark_/Documents/02_Professional/03_Unil/SATW/MDPI_paper/complementary_information/figures/Fig_IFRMtest_mock_exper.svg', format="svg")
    plt.show()
    ##### Experimental bioreactor data and results
##    print('CV ref all - CV ref', (CV_ref_all[:,0]- CV_all[:,0]))
##    print('CV ref all', CV_ref_all[:,0])
##    Sr = []
##    Mr = []
##    Hr = []
##    
##    sd = 12
##    mat = 2.7
##    hdr = 1.8
##
##    ## Best R2 = 0.493 : SDF = 0.9, MF=0.3, HDF=1.1, SF=0.0018
##    
##    #SF = 0.6 # Scale factor to ajust the value predicted from every feature.  0.0018 works well for MinMax scaled data.. Not mimmax 0.6 
##
##    IFR_ref_all=IFR_1
##    
####    for i in range(len(IFR_ref_all)):
####            Sr.append(np.multiply(CV_ref_kjl[i],(
####            np.where(sd==36.0, SC1*np.sin(3.14*IFR_1[i]/Qmax), IFR_1[i])+
####            np.where(sd==36.0 and IFR_1[i] < 0.2*Qmax , SC2*np.exp(-IFR_1[i]/50), 2*SC2*IFR_1[i]) +
####            np.where(sd==12.0, SC3*np.sin(3.14*IFR_1[i]/Qmax), IFR_1[i]))*SC)*SF)
####            
####            Mr.append(np.multiply(CV_ref_kjl[i],(
####            np.where(mat==11.0 and IFR_1[i]< 0.2*Qmax, IF*np.exp(-IFR_1[i]/50), 2*IF*IFR_1[i]) +
####            np.where(mat==1.0 and IFR_1[i]> 0.8*Qmax, PVC*np.exp(-IFR_1[i]/50), PVC*IFR_1[i]))*MF)*SF)
####            
####            Hr.append(np.multiply(CV_ref_kjl[i], (
####            np.where(hdr == 0.5 and IFR_3[i]<Qmax*0.5/3.6, HDR1*IFR_3[i], IFR_3[i])+
####            np.where(hdr == 3.6 and IFR_3[i]>Qmax*0.5/3.6 and IFR_3[i]>Qmax*1.8/3.6, HDR3*IFR_3[i], IFR_3[i]))*HF)*SF)
##
##    for i in range(len(IFR_ref_all)):
##            Sr.append(np.multiply(CV_ref_kjl[i],(
##            np.where(sd==12, SC12*IFR_1[i]/Qmax, 0)+
##            np.multiply(sd,(36/36)*IFR_1[i]/Qmax))*SC)*SF)
####            np.where(exper_lit[h][0]==36.0, SC1*IFR_1[i]/Qmax, IFR_1[i]/Qmax)+
####            np.where(exper_lit[h][0]==36.0, SC2*np.sin(3.14*IFR_1[i]/Qmax), 0)+
####            np.where(exper_lit[h][0]==36.0 and IFR_1[i] < SC3*Qmax , np.exp(-IFR_1[i]/50), IFR_1[i]/Qmax) +
####            np.where(exper_lit[h][0]==4.0 and IFR_1[i] < SC4*Qmax , np.exp(IFR_1[i]/100), IFR_1[i]/Qmax) +
####            np.where(exper_lit[h][0]==4.0 and IFR_1[i] > SC3*Qmax , 0.8*IFR_1[i]/Qmax, np.exp(-IFR_1[i]/50)))*SC)*SF)
##            
##            
##            Mr.append(np.multiply(CV_ref_kjl[i],(
##            np.where(mat==2.7, MF27*IFR_1[i]/Qmax, 0)+
##            np.multiply(mat, (36/11)*IFR_1[i]/Qmax))*MF)*SF)
####            np.where(mat==11.0 and IFR_1[i]< 0.2*Qmax, IF*np.exp(-IFR_1[i]/50), 2*IF*IFR_1[i]) +
####            np.where(mat==1.0 and IFR_1[i]> 0.8*Qmax, PVC*np.exp(-IFR_1[i]/50), PVC*IFR_1[i]))*MF)*SF)
##            
##            Hr.append(np.multiply(CV_ref_kjl[i], (
##            np.where(hdr==1.8, HDR18*IFR_1[i]/Qmax, 0)+
##            np.multiply(hdr, (36/4)*IFR_1[i]/Qmax))*HF)*SF)
####            np.where(hdr == 0.5 and IFR_3[i]<Qmax*0.5/3.6, HDR1*IFR_3[i], IFR_3[i])+
####            np.where(hdr == 3.6 and IFR_3[i]>Qmax*0.5/3.6 and IFR_3[i]>Qmax*1.8/3.6, HDR3*IFR_3[i], IFR_3[i]))*HF)*SF)
##            
##    CV_red_experiment = (np.asarray(Sr) + np.asarray(Mr) + np.asarray(Hr))
##
##    test_mock = LinearRegression().fit(CV_ref_kjd,CV_red_experiment) #, fit_intercept=True
##    print("test R2 is:" + str(test_mock.score(CV_ref_kjd,CV_red_experiment)))  
##    


    fig, ax = plt.subplots()
    labels = ['S', 'M', 'H', 'Sum']
##    ax.plot(X_mock, marker='.', linestyle='')  #None
    ax.plot(S, marker='.', linestyle='')  #None
    ax.plot(M, marker='.', linestyle='')  #None
    ax.plot(H, marker='.', linestyle='')  #None
    ax.plot(IFR_Mtest, marker='.', linestyle='')  #None 
    ax.set(xlabel='Time(days)', ylabel='CV reduction (kJ/day)',
               title='Calorific value reduction per mechanical factor')
    ax.grid()
    plt.legend(labels)
    # fig.savefig("test.png")   
    plt.title('9 mock experiments (1530 days) \n Contributions of predictors, and sum of CV reduction', fontsize='large')
    #plt.savefig('C:/Users/mark_/Documents/02_Professional/03_Unil/SATW/MDPI_paper/complementary_information/figures/Fig_predictor_sum_mock_exper.svg', format="svg")
    plt.show()

##    IFR_M1_a = IFR_Mtest.reshape(3,-1)
##    print('IFR_M1_a.shape SMH x 170x9?',IFR_M1_a.shape)
##
##    IFR_M1_b = np.sum(IFR_M1_a, axis = 0)
##    print('IFR_M1_b.shape SMH x 170x9?',IFR_M1_b.shape)
    
 # # Must use fortran order to respect the original datasets
    #IFR_M1 = (np.asarray(S) + np.asarray(M) + np.asarray(H)).reshape(-1,len(exper_lit), order = 'F')
    IFR_M1 = IFR_Mtest.reshape(-1,len(exper_lit), order = 'F')
    print('IFR_M1 shape (Sum of S,M,H reshaped to make 9 experiments, 170 x 9)??:', IFR_M1.shape)
   # 

    ## MinMax scaled mock data
    scaler_mock = MinMaxScaler(feature_range=(0,1), copy=True)
    X_mock = scaler_mock.fit_transform(IFR_M1)#.astype(np.float32) 
    # plt.plot(IFR_M1)

    fig, ax = plt.subplots(figsize=(14.1, 10))
    labels = ['E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8', 'E9']#['S', 'M', 'H']
    ax.plot(X_mock[:,0], marker='.', linestyle='')  #None
    ax.plot(X_mock[:,1], marker='.', linestyle='')  #None
    ax.plot(X_mock[:,2], marker='.', linestyle='')  #None
    ax.plot(X_mock[:,3], marker='.', linestyle='')  #None
    ax.plot(X_mock[:,4], marker='.', linestyle='')  #None
    ax.plot(X_mock[:,5], marker='.', linestyle='')  #None
    ax.plot(X_mock[:,6], marker='.', linestyle='')  #None
    ax.plot(X_mock[:,7], marker='.', linestyle='')  #None
    ax.plot(X_mock[:,8], marker='.', linestyle='')  #None
##    ax.plot(H, marker='.', linestyle='')  #None
##    ax.plot(IFR_Mtest, marker='.', linestyle='')  #None 
    ax.set(xlabel='Time(days)', ylabel='Min-Max scaled CV reduction (kJ/day)',
               title='Calorific value reduction per mechanical factor, 9 mock experiments')
    ax.grid()
    plt.legend(labels)
    # fig.savefig("test.png")
##    plt.savefig('C:/Users/mark_/Documents/02_Professional/03_Unil/SATW/MDPI_paper/complementary_information/figures/Fig_cvred_9mock.svg', format="svg")
    plt.title('9 mock experiments (1530 days) \n CV reduction', fontsize='large')
    plt.show()


##    print(CV_ref_kjd)
##    print('CV_ref_kjd shape',CV_ref_kjd.shape)
##    print('CV_red_experiment shape', CV_red_experiment.shape)
    
##    plt.figure(figsize=(14.1,10))
##    labels = ['true','predicted']
##    plt.plot(CV_ref_kjd)#.iloc[:,0])
##    plt.plot(IFR_M1[:,1])#.iloc[:,0])
##    #plt.plot(CV_red_experiment)
##    plt.ylabel('CV reduction [kJ/day]')
##    plt.xlabel('sample [day]')
##    plt.legend(labels)
##    plt.title('9 mock experiments - validation test \n time series reference and predicted CV reductions', fontsize='large')
##    #plt.savefig('C:/Users/mark_/Documents/02_Professional/03_Unil/SATW/MDPI_paper/complementary_information/figures/Fig_ref_pred_time_mock_exper.svg', format="svg")
##    plt.show()


    yl = np.array([0,200]) #,0.1
    xl = np.array([0,200])
    plt.figure(figsize=(14.1,10))
    plt.plot(xl,yl,  c="k")#, linestyle='solid')
    plt.scatter(CV_ref_kjd, IFR_M1[:,1])#CV_ref_kjd, = experimental data. IFR_M1[:,1] = Cv reduction obtained using mock experiment predictors
##    plt.xlim([0,200])
##    plt.ylim([0,200])
    plt.ylabel('Mock experiment, Predicted CV reduction [kJ/day]')
    plt.xlabel('Reference experiment CV reduction [kJ/day]')
    plt.title('9 mock experiments, CV reductions \n Mock experiment with ESD=12, MAT=2.7, HDR=1.8 (IFR_M1[:,1]) versus Reference experiment (CV_ref_kjd)', fontsize='large')
    #plt.savefig('C:/Users/mark_/Documents/02_Professional/03_Unil/SATW/MDPI_paper/complementary_information/figures/Fig_ref_pred_cross_mock_exper.svg', format="svg")
    plt.show()

    
    
    # hlabels = ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9']

     ############   CV reduction rank. Sort by experiment      #################
##    y_data_s = y_data.reshape(-1, 9)
##    y_data = pd.DataFrame(data = y_data)
##    L9_y_data = y_data.reshape(-1,9, order='F')  #y_data_s.reshape
    L9_y_data = np.concatenate ([CV_ref_kjd, IFR_M1], axis=1)  # Add the reference data to a new column (10)
    Tot_describe = pd.DataFrame(L9_y_data)
    print('Descriptive statistics (reference experiment - line 1, and mock data - lines 2-10)', Tot_describe.describe().T)
    print('Medians, ref-line 1, and mock - lines 2-10',np.median(L9_y_data, axis=0).T) 

##    for a in L9_y_data: #do not need the loop at this point, but looks prettier
##        print(stats.describe(a))
##       
    
    # # Must use fortran order to respect the original datasets
    L9_sort = np.sort(L9_y_data, axis=0)
    Test = np.sum(L9_y_data[:,1:9])
    
    #Testb = np.sum(E1)
    Tot = []
    Sum_l9 = np.sum(L9_y_data, axis = 0, dtype ="float32") # The sum of each experiment
    order = Sum_l9.argsort()
    ranks = order.argsort()
    print('Descending order. Highest number = highest rank. labels : Ref experiment, 1,2,3,4,5,6,7,8,9',ranks)  # Descending order. Highest number = highest rank
##    l9_arg_descend = np.argsort(np.argsort(-Sum_l9))  #, axis=0
    l9_arg_ascend = np.argsort(-np.sum(IFR_M1, axis = 0, dtype ="float32")).argsort()
    print('Ascending order. Lowest number = highest rank. labels : Ref experiment, 1,2,3,4,5,6,7,8,9. Use argsort(-array)',l9_arg_ascend)
    L9_sort = -np.sort(-Sum_l9)
    l9_index = pd.DataFrame(data=Sum_l9).index
    Tot = pd.DataFrame(data = Sum_l9.reshape(1,10) , columns=['Experiment','one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine'])
    #print(Tot.T)
    print('CV reduction sum by experiment. labels : Ref experiment, 1,2,3,4,5,6,7,8,9', Tot.T)
    Tot_sort = []
    Tot_sort = np.sum(L9_sort, axis = 0)

    
##    plt.figure(figsize=(16,6))
##    plt.plot(Tot)
##    plt.plot(Tot_sort)
##    plt.show()
    
    ##########  Bin counts  ########
    n_bins = 15 #'auto'
    fig, axs = plt.subplots(9, 1, tight_layout=True, figsize=(10, 12), sharex=True, sharey=True)
    axs[0].set_ylim(0,40,10)
    axs[0].hist(IFR_M1[:,0], bins= n_bins)
    axs[1].hist(IFR_M1[:,1], bins= n_bins)
    axs[2].hist(IFR_M1[:,2], bins= n_bins)
    axs[3].hist(IFR_M1[:,3], bins= n_bins)
    axs[4].hist(IFR_M1[:,4], bins= n_bins)
    axs[5].hist(IFR_M1[:,5], bins= n_bins)
    axs[6].hist(IFR_M1[:,6], bins= n_bins)
    axs[7].hist(IFR_M1[:,7], bins= n_bins)
    axs[8].hist(IFR_M1[:,8], bins= n_bins)
    axs[0].set_title('Bin counts, 9 mock experiments \n 170 samples per experiment')
    axs[8].set_xlabel('CV reduction bin limits [kJ/day]')
    
    #plt.savefig('C:/Users/mark_/Documents/02_Professional/03_Unil/SATW/MDPI_paper/complementary_information/figures/Fig_mock_exper_binned.svg', format="svg")
    plt.show()

    plt.figure(figsize=(14.1,10))
    plt.plot(IFR_ref_all)#.iloc[:,0])
    plt.ylabel('Influent flow rate [l/day]')
    plt.xlabel('Sample[day]')
    plt.title('Influent flow rate, experiment')
    #plt.savefig('C:/Users/mark_/Documents/02_Professional/03_Unil/SATW/MDPI_paper/complementary_information/figures/Fig_Qin.svg', format='svg')
    plt.show()

    low = np.argwhere(IFR_M1[:,2] < 100)
    print('low',low)
    print('length IFR_ref_all',len(IFR_ref_all))
    print('length IFR_ref_all',low)

    fig, ax = plt.subplots()
    labels = ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9']#, 'Ref', 'Ref mock data']
    for j in range(len(labels)):
        ax.scatter(IFR_ref_all,IFR_M1[:,j], s=8)#, marker='.', linestyle='')  #None
        
        ax.set(xlabel='Influent flow rate [l/day]', ylabel='CV reduction (kJ/day)',
               title='Calorific value reduction vs influent flow rate \n 9 mock experiments using coefficients based on litterature')
        ax.grid()
##    ax.plot(CV_ref_kjd, marker='+', linestyle='') #  Add _mm if required.
##    ax.plot(CV_red_experiment, marker='x', linestyle='') #  ref data from mock data method
    plt.legend(labels, ncol=3, loc='upper left')
    #plt.savefig('C:/Users/mark_/Documents/02_Professional/03_Unil/SATW/MDPI_paper/complementary_information/figures/Fig_Qin_vs_CVred.svg', format='svg')
    #plt.show()

##    fig, ax = plt.subplots()
##    labels = ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9']#, 'Ref', 'Ref mock data']
##    for j in range(len(labels)):
##        ax.plot(np.sum(IFR_M1[:50,j]), marker='.', linestyle='')#, marker='.', linestyle='')  #None
####        ax.plot(np.sum(IFR_M1[51:250,j]), marker='.', linestyle='')
####        ax.plot(np.sum(IFR_M1[251:,j]), marker='.', linestyle='')
##        ax.set(xlabel='Mock experiment', ylabel='CV reduction (kJ/day)',
##               title='Day 1-50. Calorific value reduction vs influent flow rate \n 9 mock experiments using coefficients based on litterature')
##        ax.grid()
####    ax.plot(CV_ref_kjd, marker='+', linestyle='') #  Add _mm if required.
####    ax.plot(CV_red_experiment, marker='x', linestyle='') #  ref data from mock data method
##    plt.legend(labels, ncol=3, loc='upper left')
##    #plt.savefig('C:/Users/mark_/Documents/02_Professional/03_Unil/SATW/MDPI_paper/complementary_information/figures/Fig_Qin_vs_CVred.svg', format='svg')
##    plt.show()
##
##    fig, ax = plt.subplots()
##    labels = ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9']#, 'Ref', 'Ref mock data']
##    for j in range(len(labels)):
####        ax.plot(np.sum(IFR_M1[:50,j]), marker='.', linestyle='')#, marker='.', linestyle='')  #None
##        ax.plot(np.sum(IFR_M1[51:250,j]), marker='.', linestyle='')
####        ax.plot(np.sum(IFR_M1[251:,j]), marker='.', linestyle='')
##        ax.set(xlabel='Mock experiment', ylabel='CV reduction (kJ/day)',
##               title='Day 51-250. Calorific value reduction vs influent flow rate \n 9 mock experiments using coefficients based on litterature')
##        ax.grid()
####    ax.plot(CV_ref_kjd, marker='+', linestyle='') #  Add _mm if required.
####    ax.plot(CV_red_experiment, marker='x', linestyle='') #  ref data from mock data method
##    plt.legend(labels, ncol=3, loc='upper left')
##    #plt.savefig('C:/Users/mark_/Documents/02_Professional/03_Unil/SATW/MDPI_paper/complementary_information/figures/Fig_Qin_vs_CVred.svg', format='svg')
##    plt.show()
##
##    fig, ax = plt.subplots()
##    labels = ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9']#, 'Ref', 'Ref mock data']
##    for j in range(len(labels)):
####        ax.plot(np.sum(IFR_M1[:50,j]), marker='.', linestyle='')#, marker='.', linestyle='')  #None
####        ax.plot(np.sum(IFR_M1[51:250,j]), marker='.', linestyle='')
##        ax.plot(np.sum(IFR_M1[251:516,j]), marker='.', linestyle='')
##        ax.set(xlabel='Mock experiment', ylabel='CV reduction (kJ/day)',
##               title='Day 251-516. Calorific value reduction vs influent flow rate \n 9 mock experiments using coefficients based on litterature')
##        ax.grid()
####    ax.plot(CV_ref_kjd, marker='+', linestyle='') #  Add _mm if required.
####    ax.plot(CV_red_experiment, marker='x', linestyle='') #  ref data from mock data method
##    plt.legend(labels, ncol=3, loc='upper left')
##    #plt.savefig('C:/Users/mark_/Documents/02_Professional/03_Unil/SATW/MDPI_paper/complementary_information/figures/Fig_Qin_vs_CVred.svg', format='svg')
##    plt.show()
##    
##    
##    
##    fig, ax = plt.subplots()
##    labels = ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9']#, 'Ref', 'Ref mock data']
##    for j in range(len(labels)):
##        ax.plot(IFR_M1[:,j], marker='.', linestyle='')  #None
##        
##        ax.set(xlabel='Time(days)', ylabel='CV reduction (kJ/day)',
##               title='Calorific value reduction, 9 mock experiments \n Using coefficients based on litterature')
##        ax.grid()
####    ax.plot(CV_ref_kjd, marker='+', linestyle='') #  Add _mm if required.
####    ax.plot(CV_red_experiment, marker='x', linestyle='') #  ref data from mock data method
##    plt.legend(labels, ncol=3, loc='upper left')
##    # fig.savefig("test.png")
##    plt.show()

    ####  To develop297/210 = 1.414.  10 y 1.414 = 14.1
##
##     X_mock = scaler_mock.fit_transform(IFR_M1)#.astype(np.float32) 
##    fig, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(14.1,10), sharey=True, tight_layout=True, gridspec_kw={'width_ratios': [4, 1]})
##    
##    labels = ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9']#, 'Ref'] #
##    for j in range(len(labels)):  #exper_lit
##        ax1.plot(IFR_M1[:,j], marker='.', linestyle='')  #None
##        
##        ax1.set(xlabel='Time [days]', ylabel='CV reduction [kJ.day$^{-1}$]',)
##               #title='Calorific value reduction, 9 mock experiments \n Using coefficients based on litterature')
##        ax1.plot(CV_ref_kjd, marker='+', linestyle='', markersize=5, c='k') #  Add _mm if required.       
####    ax1.plot(CV_red_experiment, marker='s', linestyle='', markersize=5, c='k') #  Add _mm if required.
####    ax1.plot(IFR_M1[:,1], marker='s', linestyle='', markersize=2, c='r') #  Add _mm if required.
##    ax1.grid()
##    
##    
##    n_bins=17
##    for k in range(len(labels)):
##        ax2.hist(IFR_M1[:,k], bins= n_bins, orientation='horizontal', stacked=True)
##        # axs[1].hist(IFR_M1[:,1], bins= n_bins)
##        ax2.set(xlabel='Bin count') #ylabel='CV reduction (kJ/day)',
##                # title='Calorific value reduction, 9 mock experiments \n Using coefficients based on litterature')
##        # ax2.grid()
##    # ax1.plot(CV_ref_all, marker='+', linestyle='') #  Add _mm if required. 
##    plt.legend(labels)
##    #plt.savefig('C:/Users/mark_/Documents/02_Professional/03_Unil/SATW/MDPI_paper/MDPI_article/MDPI_article_v24/figures/F2_cv_reduction_l9.svg', format="svg")
##    plt.show()
##    

    
####  297/210 = 1.414.  10 y 1.414 = 14.1    
    fig, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(16.2,10), sharey=True, tight_layout=True, gridspec_kw={'width_ratios': [4, 1]})
    
    labels = ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9']#, 'Ref'] #
    for j in range(len(labels)):  #exper_lit
        ax1.plot(IFR_M1[:,j], marker='.', linestyle='')  #None
        ax1.tick_params(axis="y", labelsize=16, direction="in")
        ax1.set_ylabel("Calorific value reduction [J.liter$^{-1}$]", fontsize=24)
        ax1.set_xlabel("Time [days]", fontsize=24)
        #ax1.set(xlabel='Time [days]', ylabel='CV reduction [kJ.day$^{-1}$]')
               #title='Calorific value reduction, 9 mock experiments \n Using coefficients based on litterature')
        ax1.plot(CV_ref_kjd, marker='+', linestyle='', markersize=5, c='k') #  Add _mm if required.
        ax1.tick_params(axis="x", labelsize=16, direction="in")
##    ax1.plot(CV_red_experiment, marker='s', linestyle='', markersize=5, c='k') #  Add _mm if required.
##    ax1.plot(IFR_M1[:,1], marker='s', linestyle='', markersize=2, c='r') #  Add _mm if required.
    ax1.grid()
    
    
    n_bins=17
    for k in range(len(labels)):
        ax2.hist(IFR_M1[:,k], bins= n_bins, orientation='horizontal', stacked=True)
        # axs[1].hist(IFR_M1[:,1], bins= n_bins)
        ax2.set_xlabel("Bin count", fontsize=24)
        #ax2.set(xlabel='Bin count') #ylabel='CV reduction (kJ/day)',
                # title='Calorific value reduction, 9 mock experiments \n Using coefficients based on litterature')
        # ax2.grid()
    # ax1.plot(CV_ref_all, marker='+', linestyle='') #  Add _mm if required.
        ax2.tick_params(axis="x", labelsize=16, direction="in")
    plt.legend(labels, fontsize=12)
    #plt.savefig('C:/Users/mark_/Documents/02_Professional/03_Unil/SATW/MDPI_paper/Journal communication/Sustainability/Round two special issue/Figures_originals/F2_cv_reduction_l9.svg', format="svg")
    plt.savefig('C:/Users/mark_/Documents/02_Professional/03_Unil/SATW/MDPI_paper/Journal communication/Sustainability/Round two special issue/Figures_originals/F2_cv_reduction_l9.svg', format="svg", bbox_inches="tight", dpi=300)
    plt.savefig('C:/Users/mark_/Documents/02_Professional/03_Unil/SATW/MDPI_paper/Journal communication/Sustainability/Round two special issue/Figures_originals/F2_cv_reduction_l9.pdf', format="pdf", bbox_inches="tight", dpi=300)
    plt.show()
    
    y_data = IFR_Mtest.reshape(-1,1)
    
    return  y_data, X_data, CV_ref_kjd, IFR_ref_all, IFR_M1
   ##End : NN_QX_mockdata()



#### Surrogate data from 4 bioreactors operated in parallel. Identical influent flow. 3 mechanical set points according to L4 or L9 plan.
def NN_LX_traindata():
    # Read Excel file contaning the new (December 2020) "Reference experiment" CV reduction data and original (same) flow rate data and the L4 experiment data
    # RD = pd.read_excel('C:\\Users\\mark_\\Anaconda3\Data\SATW_project.xlsx', sheet_name='Experiment_L9_V3') #sep=";",,  decimal=','Experiment_L4_V2
    RD = pd.read_excel('C:\\Users\\mark_\\Anaconda3\Data\SATW_project.xlsx', sheet_name='Mock experiment_E9')
    # RD = pd.read_excel('/scratch/mmccormi1/SATW_project.xlsx', sheet_name='Mock experiment_E9', index_col=None, engine="openpyxl") #sep=";",,  decimal=','Experiment_L4_V2
    # RD = np.genfromtxt(open("/scratch/mmccormi1/SATW_project.csv"), delimiter=';')
    RD_Ex = pd.DataFrame(data = RD) # RD_Ex = reference data from the experiment with additional data derived from T and Q (170 data points) plus mock data created using equation X
    
    CV_ref_all = np.multiply(RD_Ex.iloc[:170,[10]], RD_Ex.iloc[:170,[7]])  #  ref -> CV reduction derived from the "Experiment" (l/day x kJ/l = kJ/day). RD_Ex.iloc[7:226,38]
    # np.savetxt("C:/Users/mark_/userdata/Output/CV_ref_all.csv", CV_ref_all, delimiter=',')
    CV_ref_all.plot()
    IFR_ref_all = RD_Ex.iloc[:170,[7]] # ref -> Reference "Experiment". Real influent flow rate data (l/day)
    # np.savetxt("C:/Users/mark_/userdata/Output/Q_influent_exp.csv", IFR_ref_all, delimiter=',')
    # np.savetxt("C:/Users/mark_/Anaconda3/Data/Q_influent_exp.csv", IFR_ref_all, delimiter=',')
    # quantile_transformer = QuantileTransformer(output_distribution='normal', random_state=0) #preprocessing.
    
    
    # Surrogate CV reduction data used in the L9 experiments. Remove lines containg zero.14  
    E1 = RD_Ex.iloc[:170,14]# kJ/day   [7:226,20].dropna() Experiments_L9_V3: 20, 22, 24, 26, 28, 30, 32, 34, 36
    # E1 = E1.loc[(IFR_ref_all != 0)]# fillna(0)
    E2 = RD_Ex.iloc[:170,16]# [7:226,22].dropna()
    # E2 = E2.loc[(IFR_ref_all != 0)]
    E3 = RD_Ex.iloc[:170,18]#[7:226,24].dropna()
    #E3 = E3.loc[(IFR_ref_all != 0)]
    E4 = RD_Ex.iloc[:170,20]#[7:226,26].dropna()
    #E4 = E4.loc[(IFR_ref_all != 0)]
    E5 = RD_Ex.iloc[:170,22]#[7:226,26].dropna()
    E6 = RD_Ex.iloc[:170,24]#[7:226,26].dropna()
    E7 = RD_Ex.iloc[:170,26]#[7:226,26].dropna()
    E8 = RD_Ex.iloc[:170,28]#[7:226,26].dropna()
    E9 = RD_Ex.iloc[:170,30]#[7:226,26].dropna()
    
    col_names=["E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8", "E9", "ref_exper"]
    # l9_CVred = np.reshape(np.concatenate([CV_ref_all.iloc[:170,0], E1, E2, E3, E4, E5, E6, E7, E8, E9], axis=0), (-1, 10))#,RD_Ex5,RD_Ex6,RD_Ex7,RD_Ex8,RD_Ex9
    # l9_CVred = np.concatenate([CV_ref_all.iloc[:170,0].to_numpy(), E1, E2, E3, E4, E5, E6, E7, E8, E9], axis=1)#
    frames = [E1, E2, E3, E4, E5, E6, E7, E8, E9, CV_ref_all.iloc[:170,0]]
    
    plt.plot(np.asarray(frames).T)
    plt.xlabel('Day number')
    plt.ylabel('Calorific value reduction [kJ/day]')
    plt.xlim([0,225])
    label=col_names
    plt.legend(label, loc=7, frameon=False)
    plt.title('9 mock experiments - what?', fontsize='large')
##    plt.show()
    
    l9_CVred = pd.concat(frames, axis=1)# Data used to make figure 2
    # l9_CVred.describe()
    l9_CVred.plot(marker='+')
    # print(CV_ref_all.iloc[:170,0])
    # np.savetxt("C:/Users/mark_/userdata/Output/l9_cvred.csv", l9_CVred, delimiter=';', header=';'.join(col_names))
    
    x = np.arange(1,171,1)
    
    print(np.amax(CV_ref_all))
    print(l9_CVred.describe())
    L9_desc_stat=l9_CVred.describe().transpose()
    # L9_skew = pd.DataFrame(scipy.stats.skew(l9_CVred, axis=0, bias=True, nan_policy='propagate'))
    # L9_kurtosis = pd.DataFrame(scipy.stats.kurtosis(l9_CVred, axis=0, fisher=True, bias=True, nan_policy='propagate'))
    L9_skew = l9_CVred.skew(axis=0)
    L9_kurtosis = l9_CVred.kurtosis(axis=0)
    L9_median = l9_CVred.median(axis=0)
    frame_stats= pd.concat([L9_kurtosis, L9_skew, L9_median]  , axis=1, ignore_index=False)  
    # L9_stats_CVred = frame_stats.append(L9_desc_stat,ignore_index=True, verify_integrity=True, sort=False)#, sort=True) #, L9_skew, ignore_index=True
    L9_stats_CVred = pd.concat([L9_desc_stat, frame_stats] , axis=1, ignore_index=False, verify_integrity=True, sort=True, copy=False) #, ignore_index=TrueL9_stats_CVred.merge(right=L9_skew, how='left') #, L9_skew
    L9_stats_CVred.columns = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max', 'kurtosis', 'skew', 'median'] 
    # L9_stats_CVred.columns = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max', 'kurt', 'skew']
    
    # L9_stats_CVred.to_csv("C:/Users/mark_/userdata/Output/L9_stats_CVred.csv", index=True)
    # np.savetxt("C:/Users/mark_/userdata/Output/L9_stats_CVred.csv", L9_stats_CVred, delimiter=';', header=';'.join(L9_stats_CVred.columns))
    
    print(L9_stats_CVred.describe())
    L9_stats_CVred.plot()
    L9_skew.transpose().plot()
    L9_kurtosis.transpose().plot()
   # CV_ref = CV_ref_all.loc[(IFR_ref_all != 0)]
    IFR_ref=IFR_ref_all#.loc[(IFR_ref_all != 0)]
    print(np.amax(IFR_ref_all))
    # CV_ref_a = CV_ref.reset_index(drop=True)
    ns = len(CV_ref_all)  #number of samples not containing zero. IFR_ref
    
    ######    MDPI figure 2    #########
    
    # plt.figure(figsize=(6,6))
    # label = ["Reference experiment",'E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8', 'E9']
    # plt.plot(CV_ref_all.reset_index(drop=True), linestyle=" ", marker='+') #CV_ref.reset_index(drop=True)
    # plt.plot(E1.reset_index(drop=True), linestyle=" ", marker='o')
    # plt.plot(E2.reset_index(drop=True), linestyle=" ", marker='o')
    # plt.plot(E3.reset_index(drop=True), linestyle=" ", marker='o')
    # plt.plot(E4.reset_index(drop=True), linestyle=" ", marker='o')
    # plt.plot(E5.reset_index(drop=True), linestyle=" ", marker='o')
    # plt.plot(E6.reset_index(drop=True), linestyle=" ", marker='o')
    # plt.plot(E7.reset_index(drop=True), linestyle=" ", marker='o')
    # plt.plot(E8.reset_index(drop=True), linestyle=" ", marker='o')
    # plt.plot(E9.reset_index(drop=True), linestyle=" ", marker='o')
    # plt.legend(label)
    # plt.ylabel('Calorific value reduction (kJ/day)')
    # plt.xlabel('Day number')
    # plt.title('Reference and surrogate timeseries', fontsize='large')
    # # plt.savefig('C:/Users/mark_/Documents/02_Professional/03_Unil/SATW/MDPI_paper/MDPI_article/MDPI_article_v12/figures/F3_cv_reduction_l9.svg', format="svg")
    # # plt.savefig('/scratch/mmccormi1/F3_cv_reduction_l4.svg', format="svg")
    # # plt.show()

################    MDPI figure 2   Scatter histogram of surrogate data   ###########################3
    # from mpl_toolkits.axes_grid1 import make_axes_locatable
    
    # FX_AB, axs = plt.subplots(1,2, figsize=(10 , 5), sharex=False, sharey=True)
    # Fixing random state for reproducibility
    # np.random.seed(19680801)
    
    # the random data
    # x = np.random.randn(1000)
    # y = 250 #np.random.randn(300)
    # x = np.arange(1,171,1)
    # # y = np.arange(1,171,1)
    # sz = 3
    # labels = ["Reference experiment",'E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8', 'E9']
    # fig= plt.figure(figsize=(15, 10))
    # ax = fig.add_subplot(111)
    
    # # the scatter plot:
    # ax.scatter(x, CV_ref_all.reset_index(drop=True), marker= '+', s=100)
    # ax.scatter(x, E1, marker= ',', s=sz)#.reset_index(drop=True))
    # # ax.fill_between(x, E1, facecolor='yellow', alpha=0.5)
    # ax.scatter(x, E2, marker= ',', s=sz)#.reset_index(drop=True))
    # ax.scatter(x, E3, marker= ',', s=sz)#.reset_index(drop=True))
    # ax.scatter(x, E4, marker= ',', s=sz)#.reset_index(drop=True))
    # ax.scatter(x, E5, marker= ',', s=sz)#.reset_index(drop=True))
    # ax.scatter(x, E6, marker= ',', s=sz)#.reset_index(drop=True))
    # ax.scatter(x, E7, marker= ',', s=sz)#.reset_index(drop=True))
    # ax.scatter(x, E8, marker= ',', s=sz)#.reset_index(drop=True))
    # ax.scatter(x, E9, marker= ',', s=sz)#.reset_index(drop=True))
    # # # Set aspect of the main axes.
    # # ax.set_aspect(1.)
    # # ax.set_xlim(auto = True)
    
    # ax.set_ylim(ymin= 0, ymax = 550) #), auto = False
    # ax.legend(labels)
    # ax.set_ylabel("Calorific value reduction [J/day]", fontsize = 8)
    # ax.set_xlabel('Experimental time series [Days]')
    # plt.tight_layout()
    # # plt.savefig('C:/Users/mark_/Documents/02_Professional/03_Unil/SATW/MDPI_paper/MDPI_article/MDPI_article_v14/figures/F2_a_cv_reduction_l9_legend.svg', format="svg")
    # # plt.savefig('C:/Users/mark_/Documents/02_Professional/03_Unil/SATW/MDPI_paper/MDPI_article/MDPI_article_v14/figures/F2_a_cv_reduction_l9.svg', format="svg")
    # plt.show()
    
    # create new axes on the right of the current axes
    # divider = make_axes_locatable(ax)
    # # below height and pad are in inches
    # # ax_histx = divider.append_axes("top", 1.2, pad=0.1, sharex=ax)
    # ax_histy = divider.append_axes("right", 0.5, pad=0.1, sharey=ax)
    
    # # make some labels invisible
    # # ax_histx.xaxis.set_tick_params(labelbottom=False)
    # ax_histy.yaxis.set_tick_params(labelleft=False)
    # # xticks(np.arange(0, 1, step=0.2))
    # ax_histy.set_ylim(ymin= 0, ymax = 250)
    
    # now determine nice limits by hand:
    # my_bins = [1, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250]#34#
    # my_bins = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260]#34#
    # my_bins = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260]#34#
    my_bins  = np.arange(0, 300, 25 )  # 0, 650, 50
    # my_bins =np.array([0.0, 1.0, 2.5, 4.0, 10.0]) 
    # xaxis = np.arange(0,12,1)
    xaxis = ('0','25', '50', '75', '100','125', '150', '175', '200', '225', '250' )#'50', '100', '150', '200', '250', '300', '350', '400', '450', '500', '550', , '600'  '0', 
    print(len(my_bins))
    print(len(xaxis))  # len xaxis must be len(my_bins) - 1
    # w = [1 for n in range(len(E1))]
    # ybars = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # w = 0.5  # width
    
    # xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
    # # lim = (int(xymax/binwidth) + 1)*binwidth
    hCVref = np.histogram(CV_ref_all.reset_index(drop=True), bins=my_bins, density=False)
    hE1 = np.histogram(E1, bins= my_bins, density=False) #bins=len(my_bins),
    hE2 = np.histogram(E2, bins= my_bins, density=False)#, density=True
    hE3 = np.histogram(E3, bins=my_bins, density=False)
    hE4 = np.histogram(E4, bins=my_bins, density=False)
    hE5 = np.histogram(E5, bins=my_bins, density=False)
    hE6 = np.histogram(E6, bins=my_bins, density=False)
    hE7 = np.histogram(E7, bins=my_bins, density=False)
    hE8 = np.histogram(E8, bins=my_bins, density=False)
    hE9 = np.histogram(E9, bins=my_bins, density=False)
    # print(np.sum(hE2[0]))
    # print(len(my_bins))
    # print(len(w))
    # df = pd.DataFrame({labels[0] :  np. asarray(hCVref[0]), labels[1] :  np. asarray(hE1[0]), labels[2] : np. asarray(hE2[0]), labels[3] : np. asarray(hE3[0]), labels[4] : np. asarray(hE4[0]), 
    #                     labels[5] : np. asarray(hE5[0]), labels[6] : np. asarray(hE6[0]), labels[7] : np. asarray(hE7[0]), labels[8] : np. asarray(hE8[0]), 
    #                     labels[9] : np. asarray(hE9[0])})
   
    
    # plt.plot(hE4[0])
    # plt.fill_between(xaxis,hE4[0], facecolor='#d62728', alpha=0.7)
    # L9_binstats = df.describe()
    # np.savetxt("C:/Users/mark_/Documents/02_Professional/03_Unil/SATW/MDPI_paper/MDPI_article/MDPI_article_v14/SATW_L9_stats.csv", L9_binstats, delimiter=";", fmt="%10.2f") 
    # label = ['E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8', 'E9', "Reference experiment",]
    # fig = plt.figure(figsize=(10, 10))
    # plt.scatter(np.array(my_bins), hE1[0],marker= '')
    # plt.fill_between(my_bins,hE1[0], facecolor='#1f77b4', alpha=1)
    # plt.scatter([my_bins],hE2[0],marker= '')
    # plt.fill_between(my_bins,hE2[0].T, facecolor='#ff7f0e', alpha=0.9)
    # plt.scatter(np.array(my_bins),hE3[0],marker= '')
    # plt.fill_between(my_bins,hE3[0], facecolor='#2ca02c', alpha=0.8)
    # plt.scatter([my_bins],hE4[0],marker= '')
    # plt.fill_between(my_bins,hE4[0], facecolor='#d62728', alpha=0.7)
    # plt.scatter([my_bins],hE5[0],marker= '')
    # plt.fill_between(my_bins,hE5[0], facecolor='#9467bd', alpha=0.6)
    # plt.scatter([my_bins],hE6[0],marker= '')
    # plt.fill_between(my_bins,hE6[0], facecolor='#8c564b', alpha=0.5)
    # plt.scatter([my_bins],hE7[0],marker= '')
    # plt.fill_between(my_bins,hE7[0], facecolor='#e377c2', alpha=0.4)
    # plt.scatter([my_bins],hE8[0],marker= '')
    # plt.fill_between(my_bins,hE8[0], facecolor='#7f7f7f', alpha=0.3)
    # plt.scatter([my_bins],hE9[0],marker= '')
    # plt.fill_between(my_bins,hE9[0], facecolor='#bcbd22', alpha=0.2)
    # plt.scatter([my_bins],hCVref[0],marker= ',')
    # plt.fill_between(my_bins,hCVref[0], facecolor='#17becf', alpha=0.1)  
    # plt.xlabel('Calorific value reduction [J/day]')
    # plt.ylabel('Bin count [days]')
    # # plt.legend(label)
    # plt.tight_layout()
    # # plt.savefig('C:/Users/mark_/Documents/02_Professional/03_Unil/SATW/MDPI_paper/MDPI_article/MDPI_article_v14/figures/F2_b_cv_reduction_l9_stacked.svg', format="svg")
    # plt.show()
    
    
    #######  Comment during HPC runs
    label = ['E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8', 'E9', "Reference experiment"]
    fig= plt.figure()#figsize=(10, 30))
   
    plt.plot(hE1[0],color='#1f77b4',marker= '')
    plt.fill_between(xaxis,hE1[0], facecolor='#1f77b4', alpha=1)
    plt.plot(hE2[0],color='#ff7f0e',marker= '')
    plt.fill_between(xaxis, hE2[0], facecolor='#ff7f0e', alpha=0.9)
    plt.plot(hE3[0],color='#2ca02c',marker= '')
    plt.fill_between(xaxis,hE3[0], facecolor='#2ca02c', alpha=0.8)
    plt.plot(hE4[0],color='#d62728',marker= '')
    plt.fill_between(xaxis,hE4[0], facecolor='#d62728', alpha=0.7)
    plt.plot(hE5[0],color='#9467bd',marker= '')
    plt.fill_between(xaxis,hE5[0], facecolor='#9467bd', alpha=0.6)
    plt.plot(hE6[0],color='#8c564b',marker= '')
    plt.fill_between(xaxis,hE6[0], facecolor='#8c564b', alpha=0.5)
    plt.plot(hE7[0],color='#e377c2',marker= '')
    plt.fill_between(xaxis,hE7[0], facecolor='#e377c2', alpha=0.4)
    plt.plot(hE8[0],color='#7f7f7f',marker= '')
    plt.fill_between(xaxis,hE8[0], facecolor='#7f7f7f', alpha=0.3)
    plt.plot(hE9[0],color='#bcbd22',marker= '')
    plt.fill_between(xaxis,hE9[0], facecolor='#bcbd22', alpha=0.2)
    plt.plot(hCVref[0],marker= 'o', linewidth=3, markersize=12, color='#17becf')
    plt.fill_between(xaxis,hCVref[0], facecolor='#17becf', alpha=0.1)  
   
    plt.xlabel('Calorific value reduction [J/day]')
    plt.ylabel('Bin count [days]')
    plt.legend(label)
    plt.title('9 mock experiments - distribution of CV reductions', fontsize='large')
    #plt.xlim([0,1000])
    # set the spacing between subplots
##    plt.subplots_adjust(left=0.1,
##                    bottom=0.1, 
##                    right=0.9, 
##                    top=0.9, 
##                    wspace=0.4, 
##                    hspace=0.4)
    plt.tight_layout()
    #plt.savefig('C:/Users/mark_/Documents/02_Professional/03_Unil/SATW/MDPI_paper/complementary_information/figures/Fig_cv_reduction_l9_distribution.svg', format="svg")
    #plt.show()
    ########  Comment during HPC runs
    
    # df = pd.DataFrame(np.asarray(hCVref[0]), np.asarray(hE1[0]), np.asarray(hE2[0]), np.asarray(hE3[0]), np.asarray(hE4[0]), 
    #                    np.asarray(hE5[0]), np.asarray(hE6[0]), np.asarray(hE7[0]), np.asarray(hE8[0]), 
    #                    np.asarray(hE9[0]))
    # df.plot.barh(stacked=True)
    # print(len(bins))
    # print(len(hE1[0]))
    # print(len(ybars))
# ax.bar(range(len(my_bins)-1),h, width=1), edgecolor='k')
# ax.set_xticks(range(len(my_bins)-1))
# ax.set_xticklabels(my_bins[:-1])
    # bins = np.arange(0, lim + binwidth, binwidth)  #(-lim, lim + binwidth, binwidth)
    # ax_histx.hist(x, bins=bins)
    # ax_histy.hist(CV_ref_all.reset_index(drop=True), histtype='bar',density = False, bins=bins, bottom=bottom, orientation='horizontal', label = labels[0])  # 
    # ax_histy.hist(E1, bins=bins, bottom=bottom, density = False, orientation='horizontal', label = labels[1], histtype='bar')
    # ax_histy.hist(E2, bins=bins, bottom=bottom, density = False, orientation='horizontal', label = labels[2], histtype='bar')
    # ax_histy.hist(E3, bins=bins, bottom=bottom, density = False, orientation='horizontal', label = labels[3], histtype='bar')
    # ax_histy.hist(E4, bins=bins, bottom=bottom, density = False, orientation='horizontal', label = labels[4], histtype='bar')
    # ax_histy.hist(E5, bins=bins, bottom=bottom, density = False, orientation='horizontal', label = labels[5], histtype='bar')
    # ax_histy.hist(E6, bins=bins, bottom=bottom, density = False, orientation='horizontal', label = labels[6], histtype='bar')
    # ax_histy.hist(E7, bins=bins, bottom=bottom, density = False, orientation='horizontal', label = labels[7], histtype='bar')
    # ax_histy.hist(E8, bins=bins, bottom=bottom, density = False, orientation='horizontal', label = labels[8], histtype='bar')
    # ax_histy.hist(E9, bins=bins, bottom=bottom, density = False, orientation='horizontal', label = labels[9], histtype='bar')
    
    # x = np.arange(len(labels))  # the label locations
    # widthb = 0.35  # the width of the bars
    
    
    # ax_histy.barh(y=np.asarray(my_bins)-widthb/2, height= np.array(hE1[0]), width=w,  align = 'edge') # range(len(bins)-1)range(len(bins)-1)
    # ax_histy.barh(y=np.asarray(my_bins)-widthb/2, height= np.array(hE2[0]), width=w,  align = 'edge') # range(len(bins)-1)range(len(bins)-1)
    # ax_histy.barh(y=np.asarray(my_bins)-widthb/2, height= np.array(hE3[0]), width=w,  align = 'edge') # range(len(bins)-1)range(len(bins)-1)
    # ax_histy.barh(y=np.asarray(my_bins)-widthb/2, height= np.array(hE4[0]), width=w,  align = 'edge') # range(len(bins)-1)range(len(bins)-1)
    # ax_histy.barh(y=np.asarray(my_bins)-widthb/2, height= np.array(hE5[0]), width=w,  align = 'edge') # range(len(bins)-1)range(len(bins)-1)
    # ax_histy.barh(y=np.asarray(my_bins)-widthb/2, height= np.array(hE6[0]), width=w,  align = 'edge') # range(len(bins)-1)range(len(bins)-1)
    # ax_histy.barh(y=np.asarray(my_bins)-widthb/2, height= np.array(hE7[0]), width=w,  align = 'edge') # range(len(bins)-1)range(len(bins)-1)
    # ax_histy.barh(y=np.asarray(my_bins)-widthb/2, height= np.array(hE8[0]), width=w,  align = 'edge') # range(len(bins)-1)range(len(bins)-1)
    # ax_histy.barh(y=np.asarray(my_bins)-widthb/2, height= np.array(hE9[0]), width=w,  align = 'edge') # range(len(bins)-1)range(len(bins)-1)
    
    # ax_histy.barh(y=my_bins, height= df.iloc[:,0], width=w,  align = 'edge') # range(len(bins)-1)range(len(bins)-1)
    # ax_histy.barh(y=my_bins, height= df.iloc[:,1], width=w,  align = 'edge') # range(len(bins)-1)range(len(bins)-1)
    # ax_histy.barh(y=my_bins, height= df.iloc[:,2], width=w,  align = 'edge') # range(len(bins)-1)range(len(bins)-1)
    # ax_histy.barh(y=my_bins, height= df.iloc[:,3], width=w,  align = 'edge') # range(len(bins)-1)range(len(bins)-1)
    # ax_histy.barh(y=my_bins, height= df.iloc[:,4], width=w,  align = 'edge') # range(len(bins)-1)range(len(bins)-1)
    # ax_histy.barh(y=my_bins, height= df.iloc[:,5], width=w,  align = 'edge') # range(len(bins)-1)range(len(bins)-1)
    # ax_histy.barh(y=my_bins, height= df.iloc[:,6], width=w,  align = 'edge') # range(len(bins)-1)range(len(bins)-1)
    # ax_histy.barh(y=my_bins, height= df.iloc[:,7], width=w,  align = 'edge') # range(len(bins)-1)range(len(bins)-1)
    # ax_histy.barh(y=my_bins, height= df.iloc[:,8], width=w,  align = 'edge') # range(len(bins)-1)range(len(bins)-1) 
    
    # the xaxis of ax_histx and yaxis of ax_histy are shared with ax,
    # thus there is no need to manually adjust the xlim and ylim of these
    # axis.
    # ax_histy.set_yticks([0, 250])
    # ax_histy.barh(10,) #, align = 'edge'
    # axb = fig.add_subplot(112)
    # axs[1]=df.plot.barh(stacked=True, legend=None)
    # axs[1].barh(my_bins, df, stacked=True, legend=None) # ax = ax_histy,, width=25
    # axs[0].set_ylabel("Cumulative calorific value reduction (Min-Max scaled)", fontsize = 8)
    # ax.set_xlabel([label])
    # ax_histy.legend(loc ='upper right', fontsize = 5, markerscale = 0.2)
    # ax_histx.set_yticks([0, 0.50, 1.00])
    # ax_histy.set_xticks([0, 0.5, 1.0])
    
    # fig= plt.figure(figsize=(10, 10))
    # df.plot.barh(stacked=True) # ax = ax_histy,, width=25
    # plt.tight_layout()
    # # plt.savefig('C:/Users/mark_/Documents/02_Professional/03_Unil/SATW/MDPI_paper/MDPI_article/MDPI_article_v14/figures/F2_b_cv_reduction_l9_stacked.svg', format="svg")
    # plt.show()
    
     
    # fig= plt.figure(figsize=(10, 10))
    # df.plot.barh(stacked=True, legend=None) # ax = ax_histy,, width=25
    # plt.tight_layout()
    # plt.savefig('C:/Users/mark_/Documents/02_Professional/03_Unil/SATW/MDPI_paper/MDPI_article/MDPI_article_v14/figures/F2_b_cv_reduction_l9_stacked_legend.svg', format="svg")
    # plt.show()
    
    # print(df.iloc[:,3])
    # print(hE8[0])
    # plt.barh(my_bins, df, stacked=True, legend=None)
    

# # slice the mechanical set-points from the Excel table by experiment
    #ns = 121  #number of samples
    RD_Ex1 = RD_Ex.iloc[56,2:5].to_numpy(copy=True).repeat(repeats = [ns])
    ME1r = np.reshape(RD_Ex1, (3, ns)).transpose()#np.repeat(RD_Ex.iloc[31,2:5], repeats = [223], axis=0) #2:5
    RD_Ex2 = RD_Ex.iloc[57,2:5].to_numpy(copy=True).repeat(repeats = [ns])
    ME2r = np.reshape(RD_Ex2, (3, ns)).transpose()
    RD_Ex3 = RD_Ex.iloc[58,2:5].to_numpy(copy=True).repeat(repeats = [ns])
    ME3r = np.reshape(RD_Ex3, (3, ns)).transpose()
    RD_Ex4 = RD_Ex.iloc[59,2:5].to_numpy(copy=True).repeat(repeats = [ns])
    ME4r = np.reshape(RD_Ex4, (3, ns)).transpose()
    RD_Ex5 = RD_Ex.iloc[60,2:5].to_numpy(copy=True).repeat(repeats = [ns])
    ME5r = np.reshape(RD_Ex5, (3, ns)).transpose()
    RD_Ex6 = RD_Ex.iloc[61,2:5].to_numpy(copy=True).repeat(repeats = [ns])
    ME6r = np.reshape(RD_Ex6, (3, ns)).transpose()
    RD_Ex7 = RD_Ex.iloc[62,2:5].to_numpy(copy=True).repeat(repeats = [ns])
    ME7r = np.reshape(RD_Ex7, (3, ns)).transpose()
    RD_Ex8 = RD_Ex.iloc[63,2:5].to_numpy(copy=True).repeat(repeats = [ns])
    ME8r = np.reshape(RD_Ex8, (3, ns)).transpose()
    RD_Ex9 = RD_Ex.iloc[64,2:5].to_numpy(copy=True).repeat(repeats = [ns])
    ME9r = np.reshape(RD_Ex9, (3, ns)).transpose()
    
    
    
    
#concatenate to place the L4 experiments on a single series (along the x-axis)
    M_data = np.concatenate([ME1r, ME2r, ME3r, ME4r, ME5r, ME6r, ME7r, ME8r, ME9r], axis=0)#     ,RD_Ex5,RD_Ex6,RD_Ex7,RD_Ex8,RD_Ex9
    M_data = pd.DataFrame(data = M_data)
    Q_data = np.concatenate([IFR_ref, IFR_ref, IFR_ref, IFR_ref, IFR_ref, IFR_ref, IFR_ref, IFR_ref, IFR_ref], axis=0) #     , IFR_ref, IFR_ref, IFR_ref, IFR_ref
    Q_data = pd.DataFrame(data = Q_data)
    
    X_data = pd.concat([M_data, Q_data], axis=1, join='inner') # Q_data
    
    ## Creat a new list of predictors for use in loo replicates
    # ES1 = np.concatenate([ME1r,IFR_ref], axis=1) 
    # ES2 = np.concatenate([ME2r,IFR_ref], axis=1) 
    # ES3 = np.concatenate([ME3r,IFR_ref], axis=1) 
    # ES4 = np.concatenate([ME4r,IFR_ref], axis=1) 
    # ES5 = np.concatenate([ME5r,IFR_ref], axis=1) 
    # ES6 = np.concatenate([ME6r,IFR_ref], axis=1) 
    # ES7 = np.concatenate([ME7r,IFR_ref], axis=1) 
    # ES8 = np.concatenate([ME8r,IFR_ref], axis=1) 
    # ES9 = np.concatenate([ME9r,IFR_ref], axis=1) 
   
    # Pred_loo = [ES1, ES2, ES3, ES4, ES5, ES6, ES7, ES8, ES9]
    
    plt.figure(figsize=(14.1, 10))
    label_feature = ["ESD", "MAT", "HTD", "Inflow"]
    plt.plot(X_data)
    plt.legend(label_feature)
    plt.title('Features, Full data', fontsize='large')
    # plt.savefig('/scratch/mmccormi1/Raw_features_l9.svg', format="svg")
##    plt.show()
    
    y_data = np.concatenate([E1, E2, E3, E4, E5, E6, E7, E8, E9], axis=0) #   RD_Ex.iloc[7:,17],RD_Ex.iloc[7:,21],RD_Ex.iloc[7:,25],RD_Ex.iloc[7:,29],RD_Ex.iloc[7:,33]
   

    
    ############   CV reduction rank. Sort by experiment      #################
    y_data_s = y_data.reshape(-1, 9)
    y_data = pd.DataFrame(data = y_data)
    L9_y_data = y_data.reshape(-1,9, order='F')  #y_data_s.reshape
    L9_y_data = np.concatenate ([L9_y_data, CV_ref_all], axis=-1)  # Add the reference data to a new column (10)
    # # Must use fortran order to respect the original datasets
    L9_sort = np.sort(L9_y_data, axis=0)
    Test = np.sum(L9_y_data[:,8])
    Testb = np.sum(E1)
    Tot = []
    Sum_l9 = np.sum(L9_y_data, axis = 0, dtype ="float32") # The sum of each experiment
    order = Sum_l9.argsort()
    ranks = order.argsort()
    print(ranks)  # Descending order. Highest number = highest rank
    l9_arg_descend = np.argsort(np.argsort(Sum_l9))  #, axis=0
    print(l9_arg_descend)
    L9_sort = -np.sort(-Sum_l9)
    l9_index = pd.DataFrame(data=Sum_l9).index
    Tot = pd.DataFrame(data = Sum_l9.reshape(1,10) , columns=['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'experiment'])
    
    Tot_sort = []
    Tot_sort = np.sum(L9_sort, axis = 0)
       
##    plt.figure(figsize=(16,6))
##    plt.plot(Tot)
##    plt.plot(Tot_sort)
##    plt.show()
    
    # height = 5
    # plt.bar(Sum_l9.T, height)
    
# fig, ax = plt.subplots()

# # Example data
# people = ('Tom', 'Dick', 'Harry', 'Slim', 'Jim')
# y_pos = np.arange(len(people))
# performance = 3 + 10 * np.random.rand(len(people))
# error = np.random.rand(len(people))

# ax.barh(L9_y_data, performance, xerr=error, align='center')
# ax.set_yticks(y_pos)
# ax.set_yticklabels(people)
# ax.invert_yaxis()  # labels read top-to-bottom
# ax.set_xlabel('Performance')
# ax.set_title('How fast do you want to go today?')

# plt.show()
# plt.plot(L9_y_data) 
# plt.plot(L9_sort)    
# plt.hist(L9_y_data[:,8], bins = 50)
    
    ## Normalized transformed data
    # transformer_x = Normalizer().fit(X_data)
    # X_trans = transformer_x.transform(X_data)
    # plt.plot(X_trans) 
    # plt.legend(label_feature)
    # plt.title('Features, Normalizer transform of Full data', fontsize='large')
    # plt.savefig('/scratch/mmccormi1/transformed_data_l9.svg', format="svg")
    # plt.show()
    
    # transformer_y = Normalizer().fit(y_data)
    # y_trans = transformer_y.transform(y_data)
    # plt.plot(y_trans) 
    # # plt.legend(label_feature)
    # plt.title('Target, Normalizer transform of Full data', fontsize='large')
    # plt.show()
    
    ## MinMax scaled transformed data
    # scaler = MinMaxScaler(feature_range=(0,1), copy=True)
    # X_trans = scaler.fit_transform(X_data).astype(np.float32)
    # # X_trans[:,2] =  np.multiply(X_trans[:,2], 3)
    # plt.plot(X_trans) 
    # plt.legend(label_feature)
    # plt.title('Features, MinMax scaler transform of Full data', fontsize='large')
    # plt.show()
    
    # Q_trans = quantile_transformer.fit_transform(Q_data)
    # Q_trans = pd.DataFrame(data = Q_trans)
    # plt.hist(Q_trans, bins = 50)
    # plt.plot(Q_trans)
    # plt.plot(M_data)

    # y_trans = quantile_transformer.fit_transform(y)
    # plt.hist(y_trans, bins = 50)
    # plt.plot(y_trans)

    # apply the min-max scaling in Pandas using the .min() and .max() methods. https://towardsdatascience.com/data-normalization-with-pandas-and-scikit-learn-7c1cc6ed6475
    #X_data_mm[column] = (X_data_sel[column] - X_data_sel[column].min()) / (X_data_sel[column].max() - X_data_sel[column].min())


    # copy the dataframe
    X_data_mm = np.empty([])    
    X_data_sel = X_data.iloc[:,0:4]  #Select X_data for min-max scaling. X_data.copy()
    X_data_sel.columns = [0,1,2,3]
    # scaler = MinMaxScaler()
    # print(scaler.fit(X_data_sel))
    # MinMaxScaler()
    # print(scaler.data_max_)
    # print(scaler.data_min_)
    # X_data_mm = scaler.fit_transform(X_data_sel, y=None)
    
   
    ## Archives
    # M_data_mm_L = []
    
    # RD_Ex2 = RD_Ex.iloc[57,2:5].to_numpy(copy=True).repeat(repeats = [ns])
    # M1 = P_mm[0].repeat(repeats=[len(IFR_ref_all)])
    # M2 = P_mm[1].repeat(repeats=[len(IFR_ref_all)])
    # M3 = P_mm[2].repeat(repeats=[len(IFR_ref_all)])
    # M4 = P_mm[3].repeat(repeats=[len(IFR_ref_all)])
    # ME2r = np.reshape(RD_Ex2, (3, ns)).transpose()
    # ME2r = np.reshape(RD_Ex2, (3, ns)).transpose()
    # for i in range(len(P_mm)):
    #     for j in range(len(IFR_ref_all)):
    #        M_data_mm_L.append(P_mm[i])# = np.concatenate([ME1r, ME2r, ME3r, ME4r, ME5r, ME6r, ME7r, ME8r, ME9r], axis=0)#
           
    # M_data_mm_T = np.asarray(M_data_mm_L).reshape((-1,9), order='F')
    # M_data_mm_T2 =np.split(M_data_mm_T, 3, axis=1)
    # M_data_mm_T2 = np.asarray(M_data_mm_T).reshape((-1,3), order='C')
    # M_data_mm = np.concatenate(M_data_mm_T2, axis = 0)
  
    
   
    # return X_data, y_data#, X_data_mm   y_data_mm, y_data_t, y_data_1p, y_data_yj, l9_CVred, CV_ref_all # , y_data_mm
    #return X_data_mm, y_data_mm, ME1r, ME2r, ME3r, ME4r, ME5r, ME6r, ME7r, ME8r, ME9r, IFR_ref#, ns, CV_ref_all, IFR_ref_all # For loo, run one time and then comment the return line
    # return X_data, y_data, ns, CV_ref_all, IFR_ref_all
    ###Split data to obtain training and test data
    return
    ## End:NN_LX_traindata() 

def NN_preprocess(X_data, y_data):  #,X_data_mm, y_data_mm
    # X_data_mm = pd.read_csv('C:/Users/mark_/userdata/Output/X_data_mm.csv', sep=',', decimal='.', header=None, index_col=False)
    # y_data_mm = pd.read_csv('C:/Users/mark_/userdata/Output/y_data_mm.csv', sep=',', decimal='.', header=None, index_col=False)
    X_data_mm = []
        # apply min-max scaling
    for column in range(4): #df_norm.columns
        X_data_mm.append([(X_data[:,column] - X_data[:,column].min()) / (X_data[:,column].max() - X_data[:,column].min())])
    
    X_data_mm = np.vstack(X_data_mm).T
    # print(X_data_mm)
    # X_data_mm.columns = ['1', '2', '3', '4']
    
    y_data_mm_r = []   # apply min-max scaling to raw data. y_data not transformed before min-max scaling
    y_data_r_mm = (y_data[:] - y_data[:].min()) / (y_data[:].max() - y_data[:].min())
    # np.savetxt("C:/Users/mark_/userdata/Output/y_data_mm.csv", y_data_mm, delimiter=',')
    # c = y_data_z_mm - y_data_mm
    # d = -(y_data_f - y_data)
    
    ## Standard scaler transformation (z-scores)
    
    # y_data_f = scaler.fit(y_data)
    # scaler = StandardScaler().fit(y_data)
    
    scalerz = StandardScaler()  #Standardize features by removing the mean and scaling to unit variance. z = (x - u) / s = the basic z-score formula.
    # a = scalerz.fit(y_data)
    # print(scalerz.mean_)
    # print(scalerz.transform(y_data))
    # print(scalerz.get_params(deep=True))
    y_data_z = scalerz.fit_transform(y_data)
    scalerm = MinMaxScaler()
    y_data_z_mm = scalerm.fit_transform(y_data_z)
    
    # b = np.subtract(y_data_mm_r, y_data_z_mm)
    # plt.plot(b)
    # y_data_z = scipy.stats.zscore(y_data, axis=None)  # Alternative method to z-scale
    # # scaler.mean_
    # # scaler.scale_
    # # a = y_data_t[:].min()
    # # b = y_data_t.min()
    # # plt.plot(y_data_z_mm.mean(axis=1))
    # # plt.plot(y_data_z_mm.std(axis=1))
    # y_data_z_mm_b = (y_data_z[:] - y_data_z.min()) / (y_data_z.max() - y_data_z.min())
    # # plt.figure()
    # # plt.plot(y_data_z_mm)
    # # plt.plot(y_data_z_mm_b)
    # # plt.show()
    # c = np.subtract(y_data_z_mm_b, y_data_z_mm)
    # plt.plot(c)
    
    
    # plt.figure(figsize=(16,6))
    # plt.plot(y_data)
    # plt.plot(d)
    # plt.show()
    # y_data log transformed
    y_data_t = np.log(y_data)  #log1p
    y_data_t_mm = (y_data_t[:] - y_data_t[:].min()) / (y_data_t[:].max() - y_data_t[:].min())
    # np.savetxt("C:/Users/mark_/userdata/Output/y_data_t_mm.csv", y_data_t_mm, delimiter=',')

    # y_data log(1+y) transformed
    y_data_1p = np.log1p(y_data)  
    y_data_1p_mm = (y_data_1p[:] - y_data_1p[:].min()) / (y_data_1p[:].max() - y_data_1p[:].min())
    # np.savetxt("C:/Users/mark_/userdata/Output/y_data_t_mm.csv", y_data_t_mm, delimiter=',')
    
    #  y_data Yeo-Johnson transformed
    y_data_yj = sklearn.preprocessing.power_transform(y_data.reshape(-1,1), method='yeo-johnson', standardize=True, copy=True)  # box-cox
    y_data_yj_mm = (y_data_yj[:] - y_data_yj[:].min()) / (y_data_yj[:].max() - y_data_yj[:].min())
    # y_data_yj_mm = y_data_yj_mm.reshape((-1,9), order='F') 
    
    
    
    
    # rs = np.mean(y_data_t_mm)-0.5  # amount of right shift
    

##    plt.figure(figsize=(16,6))
##    plt.xlabel('Min-Max scaled calorific value reduction [J/day]')
##    plt.ylabel('Bin count [days]')
##    Ctsy, biny, ignored = plt.hist(y_data_r_mm, 60, density=False,  label= "Raw response data")
##    Ctsyz, binyz, ignored = plt.hist((y_data_z_mm), 60, density=False, alpha=0.5, label= "z-scaled response data")
##    Ctsyt, binyt, ignored = plt.hist((y_data_t_mm), 30, density=False, alpha=0.5, label= "log transformed response data")
##    Ctsy1p, biny1p, ignored = plt.hist((y_data_1p_mm), 30, density=False, alpha=0.5, label= "log(1+y) transformed response data")
##    Ctsybc, binybc, ignored = plt.hist((y_data_yj_mm), 30, density=False, alpha=0.5, label= "Yeo-Johnson transformed response data")
##    # mu, sigma = 0.5, 0.125 # mean and standard deviation. sigma = 0.125 in normal distribution
##    # plt.plot(biny, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (biny - mu)**2 / (2 * sigma**2) ), linewidth=2, color='g', label= "Normal distribution")
##    # plt.plot(binyt, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (binyt - mu)**2 / (2 * sigma**2) ), linewidth=2, color='g', label= "Normal distribution")  #log transformed data
##    # plt.plot(binyt+rs, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (binyt - mu)**2 / (2 * sigma**2) ), linewidth=2, color='r', label= "Right shifted normal distribution")
##    plt.xlim([0,1]) 
##    plt.legend()
##    plt.title('Transformation of Min-Max scaled response data', fontsize='large')    
##    # plt.savefig('C:/Users/mark_/Documents/02_Professional/03_Unil/SATW/MDPI_paper/MDPI_article/MDPI_article_v23/figures/Fig_data_trans.svg', format="svg")
##    plt.savefig('C:/Users/mark_/Documents/02_Professional/03_Unil/SATW/MDPI_paper/complementary_information/figures/Fig_response_data_transform.svg', format="svg")
##    plt.show()
    
   
   
    # plt.scatter(X_data, y_data)
    # print('Mechanical set points', M_data.shape, 'Predictor data', X_data.shape, 'Influent', Q_data.shape,'Biogas', y_data.shape) #
    # stdscaler=StandardScaler()# with_std=False
    # X_data_mm = stdscaler.fit_transform(X_data).astype(np.float32)  #, copy=True
    # # X_data_mm = X_trans
    # y_data_mm = stdscaler.fit_transform(y_data).astype(np.float32)  #, copy=True, y_trans
    # y_data_mm = np.zeros(shape=(1530))
    # y_data_mm = y_data_yj_mm #y_data_z_mm #, y_data_yj_mm  y_data_1p_mm.reshape(-1,1)
    # print(y_data_mm)
    # plt.plot(y_data_mm)
    
    
#  The Quantile transformer 
    # qt = QuantileTransformer(n_quantiles=34, output_distribution="normal", random_state=0)
    # X_data_mm = qt.fit_transform(X_data_mm)
    
    y_data_mm = y_data_yj_mm # y-dtat_t_mm = log transformed data. y_data_yj_mm  # Set y_data_mm equal to the chosen transformation. y_data_mm_r

    ## Reshape for lstm nn
##    t=10  # t = time step (update frame) of the lstm model
##    X_data_mm_rs = np.reshape(X_data_mm, (-1, t, X_data_mm.shape[1]))
##    y_data_mm_rs = np.reshape(y_data_mm, (-1, t, y_data_mm.shape[1]))
## Set shuffle equal to True or to False. 70:30 = 1071+459=1530. (80:20 =   1224+306 = 1530        
    X_train_mm, X_test_mm, y_train_mm, y_test_mm = train_test_split(X_data_mm, y_data_mm, test_size=0.2, random_state= 42, shuffle = True, stratify= None)  #_rs, shuffle = True, random_state=42, stratify= None causes data to NOT be split in a stratefied manner.

    #X_train_mm, X_test_mm, y_train_mm, y_test_mm = train_test_split(X_data_mm, y_data_mm, test_size=int(468), random_state= 42, shuffle = True, stratify= None) 
    # inputs = tf.convert_to_tensor(X_train_mm)
    # inputs_test = tf.convert_to_tensor(X_test_mm)
    # outputs = tf.convert_to_tensor(y_train_mm)
    # outputs_test = tf.convert_to_tensor(y_test_mm)
    
    ########## Save SHUFFLED data   ########    
    # np.savetxt("C:/Users/mark_/userdata/Output/X_train_mm.csv", X_train_mm, delimiter=",", fmt="%10.4f") 
    # np.savetxt("C:/Users/mark_/userdata/Output/y_train_mm.csv", y_train_mm, delimiter=",", fmt="%10.4f") 
    # np.savetxt("C:/Users/mark_/userdata/Output/X_test_mm.csv", X_test_mm, delimiter=",", fmt="%10.4f") 
    # np.savetxt("C:/Users/mark_/userdata/Output/y_test_mm.csv", y_test_mm, delimiter=",", fmt="%10.4f")  
    
    # np.savetxt("/scratch/mmccormi1/X_train_mm.csv", X_train_mm, delimiter=",", fmt="%10.2f") 
    # np.savetxt("/scratch/mmccormi1/y_train_mm.csv", y_train_mm, delimiter=",", fmt="%10.2f") 
    # np.savetxt("/scratch/mmccormi1/X_test_mm.csv", X_test_mm, delimiter=",", fmt="%10.2f")
    # np.savetxt("/scratch/mmccormi1/y_test_mm.csv", y_test_mm, delimiter=",", fmt="%10.4f")
    
    #########  Save UN-SHUFFLED data    ############
    # np.savetxt("C:/Users/mark_/userdata/Output/X_train_mm_ns.csv", X_train_mm, delimiter=",", fmt="%10.4f") 
    # np.savetxt("C:/Users/mark_/userdata/Output/y_train_mm_ns.csv", y_train_mm, delimiter=",", fmt="%10.4f") 
    # np.savetxt("C:/Users/mark_/userdata/Output/X_test_mm_ns.csv", X_test_mm, delimiter=",", fmt="%10.4f") 
    # np.savetxt("C:/Users/mark_/userdata/Output/y_test_mm_ns.csv", y_test_mm, delimiter=",", fmt="%10.4f")  
    
    # np.savetxt("/scratch/mmccormi1/X_train_mm.csv", X_train_mm, delimiter=",", fmt="%10.2f") 
    # np.savetxt("/scratch/mmccormi1/y_train_mm.csv", y_train_mm, delimiter=",", fmt="%10.2f") 
    # np.savetxt("/scratch/mmccormi1/X_test_mm.csv", X_test_mm, delimiter=",", fmt="%10.2f")
    # np.savetxt("/scratch/mmccormi1/y_test_mm.csv", y_test_mm, delimiter=",", fmt="%10.4f")
    
    
        
    
    
    # print(X_train_mm.shape, y_train_mm.shape, X_test_mm.shape, y_test_mm.shape)
    # # print(train_test_split.splitting)
    # print(type(X_train_mm))

    # plt.figure(figsize=(16,6))
    # plt.plot(X_train_mm, label = "X_train_mm")
    # plt.plot(X_test_mm, label = "X_test_mm")
    # plt.plot(y_train_mm, label = "y_train_mm")
    # plt.plot(y_test_mm, label = "y_test_mm")
    # plt.legend()
    # plt.xlabel('Day')
    # plt.ylabel('Predictor and response values')
    # plt.title('All stratified K-fold split predictor and response values, MinMax scaled data', fontsize='large')
    # plt.show()
    
    # plt.plot(X_train_mm[:,0], marker='+', linestyle='None', c='g', label = "Spherical diameter")
    # plt.plot(X_train_mm[:,1], marker='+', linestyle='None', c='b', label = "Packing material")
    # plt.plot(X_train_mm[:,2], marker='+', linestyle='None', c='r', label = "Vessel Height/Diameter")
    # plt.plot(X_train_mm[:,3], marker='+', linestyle='None', c='m', label = "Daily influent flow")#, 'g-', outputs
    # plt.xlabel('Day')
    # plt.ylabel('Predictor value')
    # plt.title('All predictors, training data, MinMax scaled', fontsize='large')
    # plt.legend()
    # plt.show()
    # plt.plot(y_train_mm)#, label = "Daily CH4 produced"CH4 production
    # plt.xlabel('Day')
    # plt.ylabel('response value, MinMax scaled')
    # plt.title('Response (CV reduction),training data', fontsize='large')  #CH4 production
    # #plt.legend()
    # plt.show()
    
    # plt.plot(y_test_mm)# , label = "Daily CH4 produced" 'g-', outputs
    # plt.xlabel('Day')
    # plt.ylabel('response value, MinMax scaled')
    # plt.title('Response (CV reduction),testing data', fontsize='large') #CH4 production
    # #plt.legend()
    # plt.show()
    
    ########### Shaping the data before feeding to the DNN model ##########
    # f = X_train_mm.shape[1] #number of input features
    # print("number of input features:",f)
    # f2 = 1#y_train_mm.shape[0]
    # print("number of output features:",f2)
    # btch = -1 # b = batch size = number of samples to input before updating the model internal parameters
    # print("Batch size:",btch)
    
    # # #### Reshape the data to the form required by feed tensors
    # X_train_mm = X_train_mm.reshape(btch,f)
    # print("reshaped X_train_mm:", X_train_mm.shape)
    # y_train_mm = y_train_mm.reshape(btch,f2)
    # print("reshaped y_train_mm:", y_train_mm.shape)
    # X_test_mm = X_test_mm.reshape(btch,f)
    # print("reshaped X_test_mm:", X_test_mm.shape)
    # y_test_mm = y_test_mm.reshape(btch,f2)
    # print("reshaped y_test_mm:", y_test_mm.shape)
    
    # # inputs = np.array(X_train_mm)#[:,3:4]).shape[1][5:100,0:4]  For MLP
    # inputs = np.asarray(X_train_mm)#.astype('float32') #for 1-layer
    # # # # print(inputs)
    
    # # outputs =  np.array(y_train_mm)# [5:100]tuple(map(tuple,tuple(map(tuple, list(y_train_mm))) #tuple(tf.map_fn(tuple,y_train_mm)) #y_train_mm #
    # outputs = np.asarray(y_train_mm)#.astype('float32')
    # # # # print(outputs)#2:100,
    
    # # inputs_test = np.array(X_test_mm)#[:,3:4]).shape[1][5:100,0:4]
    # inputs_test = np.asarray(X_test_mm)#.astype('float32')
    # # # # print(inputs_test)
    
    # # outputs_test =  np.array(y_test_mm)# [5:100]tuple(map(tuple,tuple(map(tuple, list(y_train_mm))) #tuple(tf.map_fn(tuple,y_train_mm)) #y_train_mm #
    # outputs_test = np.asarray(y_test_mm)#.astype('float32')
    # # # print(outputs_test)#2:100, 
    
  
    
    # return inputs, outputs, inputs_test, outputs_test
    return X_train_mm, X_test_mm, y_train_mm, y_test_mm,  X_data_mm, y_data_mm, y_data_r_mm, y_data_z_mm, y_data_t_mm, y_data_1p_mm, y_data_yj_mm, y_data_z, y_data_t, y_data_1p, y_data_yj# y_data_mm = the transformed data., y_data_t_mm, y_data_1p_mm, y_data_yj_mm, y_data_z_mm
    ## End:NN_preprocess() 


################ Dataset built from polynomial model of experimental data  ######################
# Make polynomial model from experimental data set.
# Add additional data points by varying the polynomial coefficients


def poly_exp(y_train_mm, X_train_mm, y_test_mm, X_test_mm):
    z = np.asarray([12, 2.7, 1.8])
    X_p_exp = np.tile(z, [170, 1])
    X_exp = np.concatenate([X_p_exp, np.asarray(IFR_ref)], axis=1)
    
     # copy the dataframe
    X_exp_mm = np.empty([])    
    # X_data_sel = X_data.iloc[:,0:4]  #Select X_data for min-max scaling. X_data.copy()
    
    scaler = MinMaxScaler()
    print(scaler.fit(X_exp))
    MinMaxScaler()
    print(scaler.data_max_)
    print(scaler.data_min_)
    X_exp_mm = scaler.fit_transform(X_exp, y=None)
    
    X_exp_mm = (X_exp[:] - X_exp[:].min()) / (X_exp[:].max() - X_exp[:].min())
    
    p = [4, 12, 36, 1, 2.7, 11, 0.5, 1.8, 4]
    pred_3 = list(itertools.combinations(p,3)) #  full factorial, 3 factors x 3 levels 3^3 = 27. sphere diameter, Material activity, H/D
    pred_3 = pd.DataFrame(pred_3)
    print(pred_3)
    # com = len((perm_9))
        
    mdlr = LinearRegression().fit(X_exp,CV_ref_all)   # CV_ref_all is the experimentally determined CV reduction
    Wr_t = mdlr.coef_
    Ir_t = mdlr.intercept_
    #     print("Wr",Wr)
    #     print("Ir",Ir[0])
#print(Ir)
    s = [4, 12, 36]  # s = sphereical diameter
    m = [1, 2.7, 12]    # m = material type
    hd = [0.5, 1.8, 4]  # hd = height to diameter ratio
    # X_Q_mm = 
    y_data_n = []
    
    for i in range(len(pred_3)):  # len(y_test_mm), 475
        for k in range(len(X_train_mm)):
        
                # y_data_n.append(X_train_mm[k,0]*Wr_t[0] + X_train_mm[k,1]*Wr_t[1] + X_test_mm[k,2]*Wr_t[2] + X_test_mm[k,3]*Wr_t[3]+Ir_t) #X_test_mm[k,0]*Wr[0,0] + X_test_mm[k,1]*Wr[0,1] + X_test_mm[k,2]*Wr[0,2] + X_test_mm[k,3]*Wr[0,3]+Ir[0]
                y_data_n.append(X_train_mm[k,0]*Wr_t[0] + X_train_mm[k,1]*Wr_t[1] + X_test_mm[k,2]*Wr_t[2] + X_test_mm[k,3]*Wr_t[3]+Ir_t)
    return



    
#########  Tests used to generate factor arrays to ajust reference data

# Coefficient_test = pd.read_csv('C:/Users/mark_/Documents/02_Professional/03_Unil/SATW/MDPI_paper/MDPI_article/Data/Test_data.csv',  sep=" ", decimal = ".", index_col=0, header=0)  #.replace(',','.'), sheet_name='Surrogate_SATW_data'
# Coefficient_test = pd.DataFrame(Coefficient_test)

# Coefficient_test = pd.read_excel('C:/Users/mark_/Documents/02_Professional/03_Unil/SATW/MDPI_paper/MDPI_article/Data/Test_data.xlsx', usecols = [0] )  # index_col=0.replace(',','.'), sheet_name='Surrogate_SATW_data'
# Coefficient_test = pd.DataFrame(Coefficient_test)
# # Coefficient_test.astype(str)
# # Test = Coefficient_test.apply(lambda x: x.replace(',','.'))
# #  #, converters= Coefficient_test.replace(',','.')
# label = ['Sin(x)', 'Cos(x)', 'K*e(x)','K/e(x)', 'K/(M+x3)', 'Ke(-x)','K*e(-x*M)', '1/x']  #Coefficient_test.columns[1:-1]

# # Sin	Cos	Sinh	e	K/e	K/(M+x3)	Ke(-x)	K*e(-x*M)
# K = 5
# M = 3
# Pow = 3
# # print(Coefficient_test.columns# = []
# plt.figure(figsize=(16,6))

# plt.plot(np.sin(Coefficient_test))
# plt.plot(np.cos(Coefficient_test))
# plt.plot(np.exp(Coefficient_test)/K)
# plt.plot(K/np.exp(Coefficient_test))
# plt.plot(K/(M+Coefficient_test**Pow))
# plt.plot(K*np.exp(-Coefficient_test))
# plt.plot(K*np.exp(-Coefficient_test*M))
# plt.plot(1/Coefficient_test)
# plt.legend(label)
#     # plt.xlabel('Day')
#     # plt.ylabel('Predictor and response values')
#     # plt.title('All stratified Kfold split predictor and response values, MinMax scaled data', fontsize='large')
# plt.show()

#########################################################################
#########    Compute linear regression of the CV and temperature data
####  Fit an equation to reference experimental data
# X_mech = np.array(170*[(12, 2.7, 1.8)])
# X_ref = np.c_[X_mech,IFR_ref_all]  #Wow!, this works great.
# y_ref = CV_ref_all

def reg_T_CV():#"C:\Users\mark_\mark_data\Input\SATW_project.xlsx"Date	Δ PCS [MJ/day]	CH4 [l/day]	I_Q [l/day]	I_T_SP	I_Ti

    ## Regression of temperature and CV reduction data
    ##  This function creates the surrogate data and creates figure 1.
    ###   OLD, from Alessandro . SAVE THIS!! CV_exper = pd.read_csv('C:/Users/mark_/Documents/02_Professional/03_Unil/SATW/MDPI_paper/PCS_regression_v2.csv', sep=';', decimal=',')#, axis=1) , header=None, index_col=False
##    CV_exper = pd.read_excel('C:\\Users\\mark_\\mark_data\\Input\\SATW_project.xlsx', sheet_name='Regression', header=0, usecols= ['Date', 'Δ PCS [MJ/day]','I_Q [l/day]', 'I_Ti', 'I_TS', 'I_PCS', 'E_TS', 'E_PCS' ], nrows = 10, engine='openpyxl', parse_dates=True, decimal=',')#, axis=1) , header=None, index_col=False


    ###   The experimentally acquired data to be plotted on Figure 1.  (Surrogate data is generated below)
    CV_exper = pd.read_excel('C:\\Users\\mark_\\mark_data\\Input\\SATW_project.xlsx', sheet_name='PCS_regression', header=0, usecols= ['Date','I_Q', 'I_Ti_K', 'I_Ti', 'I_TS', 'I_PCS', 'E_TS', 'E_PCS' ], nrows = 21, engine='openpyxl', parse_dates=True, decimal=',', index_col='Date')
    print('CV_exper', CV_exper.head)

##    rng = np.random.default_rng()
##    y_noise = 0.07 * rng.normal(size=len(CV_exper))
    #print(y_noise)
    I_PCS = CV_exper.loc[:,'I_TS'].multiply(CV_exper.loc[:,'I_PCS']) #g/l x kJ/g = The experimental influent calorific value vector in kJ/l.
    E_PCS = CV_exper.loc[:,'E_TS'].multiply(CV_exper.loc[:,'E_PCS'])    #g/l x kJ/g = The experimental efffluent calorific value vector in kJ/l.
    T = CV_exper.loc[:,'I_Ti_K']# in K
    #T = CV_exper.loc[:,'I_Ti']# in °C 
    print('Influent temperature, K',T)
    y= 1000*I_PCS.sub(E_PCS)# + y_noise #The experimental calorific value reduction vector in J/l.
    #y_t= 1000*I_PCS.sub(E_PCS)# + y_noise #The experimental calorific value reduction vector in J/l.
    #y.add(y_noise, fill_value=0)
    ##y.index = CV_exper.loc[:,'Date']
    
    x = T.div(CV_exper.loc[:,'I_Q'])  # The experimental predictor vector (Influent temperature/Flow rate) [K*day/l] ,,,,[l/day] [l/day]
    print('x',x)
    Q = CV_exper.loc[:,'I_Q'].to_numpy()
##    plt.figure()
##    plt.plot(y)
##    plt.show()
    
    #y= CV_exper.multiply(CV_exper(loc[[:],2].sub(CV_exper.iloc[[4,6,9, 10, 12, 13, 16, 18, 19, 20],4]) # + y_noise

##      Save code below because it runs.    
##    Index = CV_exper.loc[[4,6,9, 10, 12, 13, 16, 18, 19, 20],'Date']
##    CV_in = 1000*(CV_exper.iloc[[4,6,9, 10, 12, 13, 16, 18, 19, 20],1].multiply(CV_exper.iloc[[4,6,9, 10, 12, 13, 16, 18, 19, 20],2]))   ## g/l * kJ/g = kJ/l * 1000 J/kJ = J/L
##    CV_out = 1000*(CV_exper.iloc[[4,6,9, 10, 12, 13, 16, 18, 19, 20],3].multiply(CV_exper.iloc[[4,6,9, 10, 12, 13, 16, 18, 19, 20],4]))   ## g/l * kJ/g = kJ/l* 1000 J/kJ = J/L 
##    
##    rng = np.random.default_rng()
##    y_noise = 0.07 * rng.normal(size=CV_in.size)
##    y= CV_in.sub(CV_out)/1000  # + y_noise  #.iloc[[4,6,9, 10, 12, 13, 16, 18, 19, 20],2].sub(CV_exper.iloc[[4,6,9, 10, 12, 13, 16, 18, 19, 20],4])
##    
##    Its = np.array([1,2,3,4,5,6,7,8,9,10])
##    Its = 1000*(CV_exper.iloc[[4,6,9, 10, 12, 13, 16, 18, 19, 20],1])
##    T = CV_exper.iloc[[4,6,9, 10, 12, 13, 16, 18, 19, 20],7]
##    Q = CV_exper.iloc[[4,6,9, 10, 12, 13, 16, 18, 19, 20],9]
##    x = T.div(Q)

    
    # I_PCS = CV_exper.iloc[[4,6,9, 10, 12, 13, 16, 18, 19, 20],2]
    # E_PCS = CV_exper.iloc[[4,6,9, 10, 12, 13, 16, 18, 19, 20],4]
    # CVred =I_PCS.sub(E_PCS)
    
    ## Basis function: CVred =  b* T/Q + C, in J/l
    ## where CVred = y = calculated from measurements, kJ/l 
    ## b = a * kBa, where a is a constant in days-1, to determine, and kB = 1.380649X10-23 J.K-1)
    ## T = the influent temperature, K
    ## Q = the influent flow rate, l/day
    ## C = a constant, J/L, to determine
    
    ##  Determine the objective function by fitting to the experimental data
    def objective(x, b, c):
        return b*x + c   # b*x[:,1] + c*x[:,2] + d*x[:,3] 
    # curve fit from the SciPy open source library
    popt, pcov = curve_fit(objective, x, y)  #popt, _ , bounds=(0, [7., 100.])
    print('pcov',pcov)
    # summarize the parameter values
    b, c = popt
    print('y = %.4f * x + %.4f' % (b, c))
    

##    x_all = pd.read_excel('C:\\Users\\mark_\\mark_data\\Input\\SATW_project.xlsx', sheet_name='Raw_SATW_data', false_values=[0], header=0, usecols= ['Date', 'I_Q', 'I_Ti'], nrows = 226, engine='openpyxl', parse_dates=True, decimal=',')
##    
##    plt.figure()
##    plt.plot(x_all)
##    plt.show()

##    y_surr = objective(x_all, b, c)
    
##    def objective_TS(X, a, b, c):   # Fit a curve to the ....
##        Its, x= X  # save, this works
##    # X= Its, x 
##        return a*Its + b*x + c   # b*x[:,1] + c*x[:,2] + d*x[:,3] ,  [:,0] + b*x[:,0]
##    # curve fit from the SciPy open source library
##    popt_ts, pcov_ts = curve_fit(objective_TS, (Its, x), y)  #popt, _ , bounds=(0, [7., 100.])
##    # summarize the parameter values
##    a, b, c = popt_ts
##    print('y = %.4f * I_TS + %.4f * x + %.4f' % (a, b, c))
    
    # print('y = %.2f * x + %.2f * x + %.2f * x + %.2f * x + %.2f' % (a, b, c, d, e))
    
    # polyref_a = objective(x, b, c)
    
    # plt.scatter(y, polyref_a)
    
    # p_sigma = np.sqrt(np.diag(pcov))
    
    # Q_n = pd.read_excel('C:\\Users\\mark_\\Anaconda3\Data\SATW_project.xlsx', sheet_name='Raw_SATW_data', usecols=[1]) 
    # T_n = pd.read_excel('C:\\Users\\mark_\\Anaconda3\Data\SATW_project.xlsx', sheet_name='Raw_SATW_data', usecols=[3]) 
    # D_n = pd.read_excel('C:\\Users\\mark_\\Anaconda3\Data\SATW_project.xlsx', sheet_name='Raw_SATW_data', usecols=[1,3], keep_default_na=False) 
    
    # Q_n.count()
    ##   The surrogate data to be plotted on Figure 1.
    ## Ceate the surrogate data set using the objective function
    ## Load Alessandro's table that includes the measured CVred, flow rates and temperatures AND his derived values (approx 22 temperatures...)
    ## This table is indexed by date. Only dates with both T and Q data are selected. 170 data points.
    All_n = pd.read_csv('C:/Users/mark_/Documents/02_Professional/03_Unil/SATW/MDPI_paper/Table_DCV.csv', sep=',', decimal='.', index_col='Date', parse_dates=True)
    #print(All_n)
    #All_n.index = All_n.iloc[:,0]
    print('All_n',All_n.head)
    print('length All_n', len(All_n))
    #Date_all = All_n.loc[:,'Date']

##    Date_all = pd.DataFrame(pd.DatetimeIndex(data=[All_n.iloc[:,0]],dtype='datetime64[ns]'), columns=['Date2'])
##    print('Date',Date_all)

##   df = pd.DataFrame(pd.DatetimeIndex(['2013-07-31', '2013-08-31', '2013-09-30', '2013-10-31',
##       '2013-11-30', '2013-12-31', '2014-01-31', '2014-02-28',
##       '2014-03-31', '2014-04-30', '2014-05-31', '2014-06-30'],
##        dtype='datetime64[ns]', freq='M'), columns=['month'])
##df.head(2) 



##df.Time = pd.to_timedelta(df.Time + ':00', unit='h')
##df.index = df.index + df.Time
##df = df.drop('Time', axis=1)
##df.index.name = 'Date'
##print (df)
##                       Value
##Date                        
##2004-05-01 00:15:00  3.58507
##2004-05-02 00:30:00  3.84625
    
    T_n = All_n.loc[:,'TC']+273.15  # Temperature in K
    print('T_n',T_n.head)
    
    Q_n = All_n.loc[:,'Q']  # Influent flow rate in l/day
    print('Q_n',Q_n.head)
    #Kb = 1.380649E-23
    M = T_n.div(Q_n)
    print('M',M)

    np.random.seed(42)
    mu, sigma = 0.5*M.mean(), 1.2*M.std() # mean and standard deviation. Y = 0.607 Ymax at mu +/- 1 std. 1/0.607 = 1.65. Aprox 95% of data is within 2 standard deviations +/-.
    print('mu',mu)
    print('sigma',sigma)
    x_n_noise = np.random.normal(0, sigma, len(M))

    plt.figure()
    plt.plot(x_n_noise)
    plt.title('x_n_noise vector included in the reference equation \n used to generate surrogate data')
##    plt.xlabel('T/Q')
##    plt.ylabel('CV reduction')
##    plt.xlim(0,0.5)
##    plt.ylim(0,0.5)
    #plt.savefig('C:/Users/mark_/Documents/02_Professional/03_Unil/SATW/MDPI_paper/complementary_information/figures/Fig_noise.svg', format="svg")
    #plt.show()
    
##    plt.figure()
##    plt.plot(np.mean(M))
##    plt.plot(np.std(M))
##    plt.title('M, mean and std')
####    plt.xlabel('T/Q')
####    plt.ylabel('CV reduction')
####    plt.xlim(0,0.5)
####    plt.ylim(0,0.5)
####    plt.savefig('C:/Users/mark_/Documents/02_Professional/03_Unil/SATW/MDPI_paper/complementary_information/figures/Fig_TQ_vs_CVred.svg', format="svg")
##    plt.show()    


##    reg_cols = [CV_exper.loc[:,'Date'],CV_exper.loc[:,'I_Q'],CV_exper.loc[:,'I_Ti'] ]
##    reg_df = pd.DataFrame(pd.concat(reg_cols, axis=1))
##
##    all_cols = [All_n.loc[:,'Date'],All_n.loc[:,'Q'],All_n.loc[:,'TC'] ]
##    all_df = pd.DataFrame(pd.concat(all_cols, axis=1))
##   
##    plt.figure()
##    #plt.plot(reg_df.loc[:,'I_Ti'], all_df.loc[:,'TC'])
##    reg_df.loc['I_Ti'].plot()
##    all_df.loc['TC'].plot()
##    plt.title('Temperature -PCS regression- and -Table_DCV-')
##    plt.show()
##
##    plt.figure()
##    plt.plot(reg_df.loc[:,'I_Q'], all_df.loc[:,'Q'])
##    #plt.plot(all_df.loc[:,'Q'])
##    plt.title('Influent -PCS regression- and -Table_DCV-')
##    plt.show()

    #x_n_noise = 0.1 * rng.normal(size=len(T_n))   #0.07
    # CV_red_n = b*(T_n.div(Q_n)) + c
    x_n = pd.DataFrame((T_n.div(Q_n) + x_n_noise))   # Obtain the surrogate predictor vector (T/Q) in K*day/l, columns=['Date', 'y, CV exper']
    print('T_n.div(Q_n) head',T_n.div(Q_n)) 
    print('x_n head',x_n) 
    #x_n.multiply(b)    
##    CV_red_n = objective(x_n, b, c)  # predicted CV reduction
##    CV_red_n = pd.DataFrame(data=CV_red_n)
    CV_red_n = pd.DataFrame([x_n.multiply(b)+c][0])  # Use the objective equation to create surrogate CV reduction data. CV_exper.loc[:,'I_TS'].multiply(CV_exper.loc[:,'I_PCS']), columns=['Date', 'CV_red_n']
    #CV_red_n.set_index(keys=Date_all.iloc[:,0]) #, inplace=True
    print('CV_red_n, surrogate points generated from fitted equation with noise',CV_red_n.head)


        
    plt.figure()
    plt.scatter(x_n, CV_red_n)  #T_n.div(Q_n)
    plt.title('CV reduction vs T/Q, surrogate data')
    plt.xlabel('T/Q')
    plt.ylabel('CV reduction')
##    plt.xlim(0,0.5)
##    plt.ylim(0,0.5)
    #plt.savefig('C:/Users/mark_/Documents/02_Professional/03_Unil/SATW/MDPI_paper/complementary_information/figures/Fig_TQ_vs_CVred.svg', format="svg")
    #plt.show()
##    plt.figure()
##    plt.plot(CV_red_n)
##    plt.show()
    
##    X= Its, x 
##    CV_red_n_ts = objective_TS(X, a, b, c)
##    CV_red_n_ts = pd.DataFrame(data=CV_red_n_ts)
##    plt.plot(CV_red_n_ts)
##    y_index = [CV_exper.loc[:,'Date']]
##    print(y_index)
    #y_df = y#pd.DataFrame(y)#, index = y_index)  #  Reference CV reduction, 10 measured values.
##    Index = CV_exper.loc[[4,6,9, 10, 12, 13, 16, 18, 19, 20],'Date']
##    y_df.set_index(keys=y_index, inplace=True)
    print('y, CV experiment',y.head)

    #, columns=['Date', 'y, CV exper', 'CV_red_n']
    frames = [y, CV_red_n]
    new = pd.concat(frames, axis=1, verify_integrity=False)
      
    #new.rename(columns = {'Date' : 'Date', '0' : 'y, CV exper', '0' : 'CV_red_n'}, inplace = True)
    print('New data frame -1 (experiment joined to surrogate)',new.head)
    #new['Date'] = pd.to_datetime(new['Date'], format='%Y-%m-%d', errors='coerce')
    new.index = pd.to_datetime(new.index, format='%Y-%m-%d', errors='coerce')
    print('New data frame -2 (experiment joined to surrogate with new index format)',new.head)
##    print('new dt',new.dtypes)
    plt.figure()
    plt.plot(new.iloc[:,0],linestyle=" ", marker='.', c="b")
    plt.plot(new.iloc[:,1],linestyle=" ", marker='.', c="r")
    plt.title('Experiment and surrogate data')
    plt.show()

##    label = ["Reference experiment",'Derived values'] 
    my_label = {"Reference experiment":"Reference experiment", "Derived values":"Derived values"}
    
##### MDPI Figure F1 (Experiment + surrogate data)
    fig, axs = plt.subplots(figsize=(16.2,10))#16.2
    
##    plt.figure(figsize=(12,10))
##    plt.plot(test['CVref'], linestyle=" ", marker='D', c="r")  # Reference experience
##    plt.plot(test['CV_n'], linestyle=" ", marker='.', c="b") # Derived values   
##    axs.plot(new[:(len(y)+1)], linestyle=" ", marker='D', c="r",  markersize=5, label=my_label["Reference experiment"])  # Reference experiencekind='scatter',ax=axs,
##    axs.plot(new[(len(y)+1):], linestyle=" ", marker='.', c="b", markersize=5, label=my_label["Derived values"])

    axs.plot(y, linestyle=" ", marker='D', c="r",  markersize=5, label=my_label["Reference experiment"])  # Reference experiencekind='scatter',ax=axs,
    axs.tick_params(axis="y", labelsize=16, direction="in")
    axs.set_ylabel("Calorific value reduction [J.liter$^{-1}$]", fontsize=24)
    
    axs.plot(CV_red_n, linestyle=" ", marker='.', c="b", markersize=5, label=my_label["Derived values"])
    axs.tick_params(axis="x", labelsize=16, labelrotation=-90, direction="in")
    axs.set_xlabel("Date", fontsize=24)#
    plt.subplots_adjust(bottom=0.175)
    #axs.xaxis.set_minor_locator(AutoMinorLocator())
    
    #ax.plot(CV_red_n.iloc[:,0], linestyle=" ", marker='.', c="b") # Derived values
    #plt.xlabel("Date", fontsize=20)
    #plt.xticks( rotation='vertical')#x, labels[::2],
    #plt.xticks(xticklabels[::10], rotation='vertical')
    # ax.set_xticks(test[::10])
    #axs.set_xticklabels(xticklabels[::10], rotation = 90)
    #plt.ylabel("Calorific value reduction [J.liter$^{-1}$]", fontsize=20)
    # plt.ylim([0.05,0.5])
    plt.legend(loc="upper right")  #, numpoints = 1
    #plt.title('Experimental results', fontsize='large')
    #plt.savefig('C:/Users/mark_/Documents/02_Professional/03_Unil/SATW/MDPI_paper/Journal communication/Sustainability/Round two special issue/Figures_originals/F1_experiment.svg', format="svg")
    plt.savefig('C:/Users/mark_/Documents/02_Professional/03_Unil/SATW/MDPI_paper/Journal communication/Sustainability/Round two special issue/Figures_originals/F1_experiment.svg', format="svg", bbox_inches="tight", dpi=300)
    plt.savefig('C:/Users/mark_/Documents/02_Professional/03_Unil/SATW/MDPI_paper/Journal communication/Sustainability/Round two special issue/Figures_originals/F1_experiment.pdf', format="pdf", bbox_inches="tight", dpi=300)
    plt.show()
##    
    # rms_poly = sqrt(mean_squared_error(y_ref, polyref_a))
    # print("Polynomial fit to experimental data rms error is: " + str(rms_poly))
    
    # reg_poly = LinearRegression().fit(y_ref, polyref_a) #, fit_intercept=True
    # print("Polynomial fit to experimental data R2 is:" + str(reg_poly.score(y_ref, polyref_a)))
    print('Q_n[0]', Q_n.iloc[:])
    print('CV_red_n[0]', CV_red_n.iloc[:,0])
    ##print('CV_red_n[1]', CV_red_n.iloc[:,1])
    CV_red_kjd = Q_n.iloc[:].multiply(CV_red_n.iloc[:,0], axis=0)/1000  # surrogate data to be compared to acquired data
    return  CV_red_kjd


#######Create SIMULATION test cases to use to select the optimal bioreactor configuration
def NN_simulationdata(perm_5):
# ns = number of samples per simulated experiment = 170.  ns, Qr
# Qr =experimental influent flow profile
##### Generate a list of all permutations of the mechanical set point values and the influent flow profile
#import itertools. Number of permutations = levels ^ factors
    #perm = list(itertools.product([4,12,36],[1,2,3],[0.5,1.8,4]))
    # perm = list(itertools.product([4,12,36],[1,2.4,12],[0.5,1.8,4])) # sphere diameter, Material activity, H/D
    # print(perm)
    #Q_Ex_sim = pd.DataFrame()

    
#### Create a long sample array of mechanical properties for use as predictor variables in the SIMULATION
    #length of sample data = 170 lines of data
    A=[]#pd.DataFrame() #columns = [0,1,2]
    mSX_A=[]#pd.DataFrame() #columns = [0,1,2]. 170 x 125 = 21250 lines. Matrix of mechanical predictors (does not include influent flow rate)
    permar = np.asarray(perm_5)#pd.DataFrame(perm_5)
##    print('permar.shape',permar.shape)
##    print('permar', permar)
    
    # n=ns
    n = 170 # 121 len(IFR_ref) 177   n = number of samples for 1 experiment
    c = 125  #number of permutations of the setpoints. L9 has 27 permutations. Full factorial design: 3 factors at 5 levels 5^3 = 125 permutations
    j=0
    i=0

    while j < (c):     
        A = permar[j,:]  # A is the data frame of the permuted mechanical values used in each sample.
##        print('A',A)
        mSX_exp =[[A] for i in range(n)]  # pd.DataFrame(data = ([A]*n))
        mSX_A = np.append(mSX_A, mSX_exp)# , axis=1)#, ignore_index=True) #replicate the influent flow vector by multiplying x the number of permutations
        j+=1
        
    mSX_A = np.asarray(mSX_A)
    mSX_Ar =  mSX_A.reshape((-1,3), order="C")  ## Use C order.
##    print('Mechanical predictors shape', mSX_Ar.shape)
    print('Mechanical predictors', mSX_Ar)
    
##    plt.figure()
##    plt.plot(mSX_Ar)
##    plt.title('Mechanical predictors')
##    plt.show()

###### Select the flow rates. Create a long sample array of inlet wastewater flow rate for use as predictor variables in the SIMULATION
    np.random.seed(42)
    Q_Ex_sim_A = np.random.uniform(127,516, size=(n,1)) # Experiment: 127 - 516.  OLD (94 to 516) Vector of random daily influent flow rates. Set min and max to the range of 94 to 516 covered by experiments.
    Q_Ex_sim_H = np.random.uniform(322,516, size=(n,1)) # Experiment: 322 - 516.  OLD (94 to 516) Vector of random daily influent flow rates. Set min and max to the range of 94 to 516 covered by experiments.
    Q_Ex_sim_L = np.random.uniform(127,321, size=(n,1))  #low range: 127-321. =>p = 0.018 difference between 1 and 4 (8 mm and 36 mm)

    Q_Ex_sim_HL = np.concatenate((Q_Ex_sim_H, Q_Ex_sim_L), axis=0)#.reshape(-1,1)
    print('Q_Ex_sim_HL', Q_Ex_sim_HL)

    scaler = MinMaxScaler(feature_range=(0,1), copy=True)
    Q_Ex_sim_A_mm= scaler.fit_transform(Q_Ex_sim_A)  # MinMax scaled complete simulation test dataset
    Q_Ex_sim_H_mm= scaler.fit_transform(Q_Ex_sim_H)  # MinMax scaled complete simulation test dataset
    Q_Ex_sim_L_mm= scaler.fit_transform(Q_Ex_sim_L)  # MinMax scaled complete simulation test dataset

    print('Q_Ex_sim_HL_mm', Q_Ex_sim_HL_mm)
    plt.figure()
    plt.plot(Q_Ex_sim_HL_mm)
    plt.title('Min-Max scaled influent flow rates, High and low ranges')
##    plt.savefig('C:/Users/mark_/Documents/02_Professional/03_Unil/SATW/MDPI_paper/complementary_information/figures/Fig_MInMax_Qin_HL.svg', format="svg")
    #plt.show()

    Q_Ex_H = Q_Ex_sim_HL_mm[:170]
    Q_Ex_L = Q_Ex_sim_HL_mm[170:]

    Q_Ex_list_A = []
    i=0
    for i in range(c):
        Q_Ex_list_A.append(Q_Ex_sim_A_mm)
        i+=1

    Q_Ex_list_H = []
    i=0
    for i in range(c):
        Q_Ex_list_H.append(Q_Ex_H)
        i+=1

    Q_Ex_list_L = []
    i=0
    for i in range(c):
        Q_Ex_list_L.append(Q_Ex_L)
        i+=1


##    print('Q_Ex_sim shape', Q_Ex_sim_df.shape)
##    print('Q_Ex_sim head', Q_Ex_sim_df.head(n=100))
##    plt.figure()
##    plt.plot(Q_Ex_sim)
##    plt.title('Influent flow, Simulation')
##    plt.show()
##    # Q_Ex_sim = pd.DataFrame(np.random.uniform(514,515, size=(n,1)))
    #print('Influent predictors shape',Q_Ex_sim.shape)
##    # np.savetxt("C:/Users/mark_/userdata/Output/Q_influent_sim.csv", Q_Ex_sim, delimiter=',')


#####  Select the influent flow range to join to the mechanical parameters. Join influent flow rate and mechanical properties to create simulation test data.       
##    rSQ_test = pd.DataFrame(np.asarray(Q_Ex_list_A).flatten())
##    rSQ_test = pd.DataFrame(np.asarray(Q_Ex_list_H).flatten())
    rSQ_test = pd.DataFrame(np.asarray(Q_Ex_list_L).flatten())
    print('rSQ_test shape', rSQ_test.shape)
    print('rSQ_test head', rSQ_test.head(n=20))

    mSX_Ar_mm= scaler.fit_transform(mSX_Ar)
    mSX_test = pd.DataFrame(data = mSX_Ar_mm)
    print('mSX_test shape', mSX_test.shape)
    print('mSX_test', mSX_test.head(n=20))
##
##    plt.figure()
##    rSQ_test.iloc[:500].plot()
##    plt.title('rSQ_test')
##    plt.show()
    
    #print(len(rSQ_test))
    #### 
    SX_test_mm = pd.concat([mSX_test, rSQ_test], axis=1, join='inner') #matrix of all predictors
    print('SX_test_mm', SX_test_mm.iloc[0:21250:170,:])
    # SX_test_mm = preprocessing.scale(SX_test, axis =0, with_mean=True, with_std=True).astype(np.float32)




##    plt.figure()
##    #labels = ['Spherical diameter','Material type','Vessel Height/Diameter'] #,'Inlet flow rate'
##    plt.plot(SX_test.iloc[:,0:3])#, 'g-'
##    #plt.legend(labels)
##    #     #plt.plot(epoch_count, test_loss, 'b-')
##    # #plt.legend(['Training Loss', 'Validation test loss']) #, 'Test Loss'
##    plt.xlabel('Day')
##    plt.ylabel('Predictor value')
##    plt.title('All mechanical predictors used in the simulation', fontsize='large')
##    #     # #plt.savefig('Anaconda3/envs/tensorflow_env/mark/Conv1d_loss.svg', dpi=None, facecolor='w', edgecolor='w',
##    #     #     orientation='portrait', papertype=None, format=None,
##    #     #     transparent=False, bbox_inches='tight', pad_inches=0.1,
##    #     #     frameon=None, metadata=None)
##    plt.show();
##
##    plt.figure()
##    #labels = ['Inlet flow rate'] #'Spherical diameter','Material type','Vessel Height/Diameter',
##    plt.plot(SX_test.iloc[:,3])#, 'g-'
##    #plt.legend(labels)
##    #     #plt.plot(epoch_count, test_loss, 'b-')
##    # #plt.legend(['Training Loss', 'Validation test loss']) #, 'Test Loss'
##    plt.xlabel('Day')
##    plt.ylabel('Predictor value')
##    plt.title('Influent flow predictors used in the simulation', fontsize='large')
##    #     # #plt.savefig('Anaconda3/envs/tensorflow_env/mark/Conv1d_loss.svg', dpi=None, facecolor='w', edgecolor='w',
##    #     #     orientation='portrait', papertype=None, format=None,
##    #     #     transparent=False, bbox_inches='tight', pad_inches=0.1,
##    #     #     frameon=None, metadata=None)
##    plt.show();
##   
##    
    # scale the simulation data in the same way as the training data   
##    scaler = MinMaxScaler(feature_range=(0,1), copy=True)
##    # SX_test_mm = preprocessing.scale(SX_test, axis =0, with_mean=True, with_std=True).astype(np.float32)
##    SX_test_mm= scaler.fit_transform(SX_test)  # MinMax scaled complete simulation test dataset
##    Q_Ex_sim_mm = scaler.fit_transform(Q_Ex_sim)

##    plt.figure()
##    labels = ['Spherical diameter','Material type','Vessel Height/Diameter'] #,'Inlet flow rate'
##    plt.plot(SX_test_mm[:,0:3])
##    plt.legend(labels)
##    #     #plt.plot(epoch_count, test_loss, 'b-')
##    # #plt.legend(['Training Loss', 'Validation test loss']) #, 'Test Loss'
##    plt.xlabel('Day')
##    plt.ylabel('Scaled predictor values')
##    plt.title('All scaled predictors used in the simulation', fontsize='large')
##    plt.show()
##  
##    plt.figure()
##    plt.plot(Q_Ex_sim)
##    plt.legend(labels)
##    #     #plt.plot(epoch_count, test_loss, 'b-')
##    # #plt.legend(['Training Loss', 'Validation test loss']) #, 'Test Loss'
##    plt.xlabel('Day')
##    plt.ylabel('Scaled predictor values')
##    plt.title('All scaled predictors used in the simulation', fontsize='large')
##    plt.show()
    #Q_Ex_sim_L
##    print('mSX_test - Mechanical simulation data shape', mSX_test.shape, 'mSQ_test - Influent simulation data shape', rSQ_test.shape, 'SX_test - All predictors simulation data shape', SX_test.shape) #
    return SX_test_mm #, Q_Ex_sim_mm  # Simulation data (Mech and Influent flow) from 1 replicate that is fed to the NN model


    
#     # plt.scatter(Qr, dSQ_test[:170])
#     #### OLD figure -time series
#     # plt.figure(figsize=(16,6))
#     # #label = ['Experiment', 'Simulation']
#     # plt.plot(Qr, linestyle='-')
#     # plt.plot(dSQ_test, linestyle='dotted') 
#     # # plt.legend(label)
#     # plt.xlabel('Day')
#     # plt.ylabel('Influent flow rate [l/day]')
#     # # plt.title('Experimental and Simulation influent flow rate profiles', fontsize='large')
#     # # plt.savefig('C:/Users/mark_/Documents/02_Professional/03_Unil/SATW/MDPI_paper/MDPI_article/MDPI_article_v12/figures/F5_q_exp_sim.svg', format="svg")
#     # plt.show()
    
#     plt.figure(figsize=(12,12))
#     #label = ['Experiment', 'Simulation']
#     plt.hist(Qr.T,bins=17, density=False, histtype='step', cumulative = True)
#     plt.hist(dSQ_test.T, bins=17, density=False, histtype='step', cumulative = True) 
#     # plt.legend(label)
#     plt.xlabel('Influent flow rate [l/day]')
#     plt.ylabel('Cumulative bin count [days]')
#     # plt.title('Experimental and Simulation influent flow rate profiles', fontsize='large')
#     # plt.savefig('C:/Users/mark_/Documents/02_Professional/03_Unil/SATW/MDPI_paper/MDPI_article/MDPI_article_v14/figures/F8_b_inf_cumul.svg', format="svg")
#     plt.show()
    
    # # n_Qr = np.negative(Qr)
    # counts, bins = np.histogram(Qr, bins = 17)
    # n_counts = np.negative(counts)
    # # plt.hist(bins[:-1], bins, weights=n_counts)
   
# ##########  Calculate 
#     bins = 17  #  Number of histogram bins = 17 (170/10 = 17)
#     Qr = IFR_ref_all  #Influent flow rate, experiments
#     counts, bins = np.histogram(Qr, bins = bins)  # Negative = experimental influent flow rate = blue bars
#     n_counts = np.negative(counts)
#     counts_dSQ, bins_dSQ = np.histogram(dSQ_test, bins = bins)
   
#     #  # plt.figure(figsize=(16,6))
#     # # #label = ['Experiment', 'Simulation']
#     # # plt.hist(Qr.T,bins=17, density=False, histtype='stepfilled')
#     # # plt.gca().invert_yaxis()
#     # # plt.hist(dSQ_test.T, bins=17, density=False, histtype='stepfilled') 
#     # # # plt.legend(label)
#     # # plt.xlabel('Influent flow rate [l/day]')
#     # # plt.ylabel('Bin count [days]')
#     # # # plt.title('Experimental and Simulation influent flow rate profiles', fontsize='large')
#     # # # plt.savefig('C:/Users/mark_/Documents/02_Professional/03_Unil/SATW/MDPI_paper/MDPI_article/MDPI_article_v12/figures/F5_q_exp_sim.svg', format="svg")
#     # # plt.show()
    
    
#     # ind = np.arange(len(Qr))
#     figX, ax = plt.subplots()
#     # dSQ_test_n = np.negative(dSQ_test)# This works, don't touch
#     # p1 = plt.hist(Qr.T,bins=17, density=False, histtype='bar', cumulative = False) # This works, don't touch
#     p1 = plt.hist(bins[:-1], bins, weights=n_counts, align='left')  #Experimental influent flow rate
#     # p2 = plt.hist(dSQ_test_n.T, bins=17, density=False, histtype='bar', cumulative = False) # This works, don't touch
#     p2 = plt.hist(bins[:-1], bins_dSQ, weights=counts_dSQ, align='left')  #Simulation influent flow rate
#     ax.set_ylabel('Bin count [days]')
#     ax.set_xlabel('Influent flow rate [l/day]')
#     # plt.savefig('C:/Users/mark_/Documents/02_Professional/03_Unil/SATW/MDPI_paper/MDPI_article/MDPI_article_v14/figures/F8_a_inf_binned.svg', format="svg")
#     plt.show()
    
    
     ######## K-S statistic    ###################### The Kolmogorov–Smirnov statistic quantifies a distance between the empirical distribution functions of 
  ##two samples. The null hypothesis is that the samples are drawn from the same distribution (in the two-sample case).
  ## See Wikipedia
    ## y_train_mm = y_test_mm_4b#np.array(range(1,1070))
    # m = len(Qr)#
    # n = len(dSQ_test) #
    # alpha = 0.001   # confidence level for rejection of the null hypothesis that samples are drawn from the same distribution.
    # D_ks = sqrt(-np.log(alpha/2)*((1+m/n)/(2*m)))  # The null hypothesis is rejected at level alpha if the calculated K-S statistic is > D_ks. 
    # print(D_ks)
    
    # outputs_k = []  # 476 train + 204 test samples = 680 samples
    # for sublist in [[i] for i in Qr]:  #Yhat_test_mm_4e
    #     for item in sublist:
    #         outputs_k.append(item)
    # outputs_test_k = []
    # for sublist in [[i] for i in dSQ_test]: #Yhat_test_mm_4f.tolist(): #Yhat_test_mm_4f :
    #     for item in sublist:
    #         outputs_test_k.append(item)
    # statistic, pvalue =ks_2samp(outputs_k, outputs_test_k, mode='auto')  #returns ks statistic and p-value.exact
    # print("D_ks value", D_ks)
    # print("K-S statistic", statistic)
    # print("K-S p-value", pvalue)
    # if statistic > D_ks:
    #     print("Reject the null hyposthesis. The training and validation datasets are different at alpha =", alpha)
    # else:
    #     print("The training and validation datasets are the same")
    
    # mSQ_test.shape
  
    
  
def KS_simulationdata(IFR_M1):   # Formerly l9_CVred 
# ns = number of samples per simulated experiment = 170.  
# Qr =experimental influent flow profile
# l9_CVred
##### Generate a list of all permutations of the L9 simulation experiments.
## Use the KS statistic to compare the experiments and find those that are not significantly different. 
#import itertools. Number of permutations = number of experiments = 9 (L9)
    # l9_CVred = pd.read_csv('C:/Users/mark_/userdata/Output/l9_CVred.csv', sep=',', decimal='.', header=None, index_col=False)#
    perm_9 = list(itertools.combinations(range(9),2)) # sphere diameter, Material activity, H/D
    perm_9 = pd.DataFrame(perm_9)
    #print('perm_9',perm_9)
    com = len((perm_9))
    l9_CVred = pd.DataFrame(IFR_M1)   #Formerly l9_CVred.iloc[:,:9]
    L9 = l9_CVred.iloc[:,:9]   # The L9 mock experiment set from which the actual experiment will be removed. 
        
    m = len(L9)#
    n = len(L9) #
    alpha = 0.01   # confidence level for rejection of the null hypothesis that samples are drawn from the same distribution.
    D_ks = sqrt(-np.log(alpha/2)*((1+m/n)/(2*m)))  # The null hypothesis is rejected at level alpha if the calculated K-S statistic is > D_ks (the test statistic). 
    print("K-S test statistic value:", D_ks)
    print("K-S alpha value:", alpha)
    # n=ns
    # n = 170 # 121 len(IFR_ref) 177   n = number of samples for 1 experiment
    # c = 125  #number of permutations of the setpoints. L9 has 27 permutations. Full factorial design: 3 factors at 5 levels 5^3 = 125 permutations
    j=0
    statistic = 0
    pvalue=0
    L9_KS_statistic=[]
    L9_KS_pvalue = []
    
    # for sublist in [[i] for i in dSQ_test]: #Yhat_test_mm_4f.tolist(): #Yhat_test_mm_4f :
    #     for item in sublist:
    #         outputs_test_k.append(item)
    # statistic, pvalue =ks_2samp(outputs_k, outputs_test_k, mode='auto')  #returns ks statistic and p-value.exact
    # print("D_ks value", D_ks)
    
    while j < (com):   
    # for i in range(len(perm_9)): 
        statistic, pvalue =ks_2samp(L9.iloc[:,perm_9.iloc[j,0]], L9.iloc[:,perm_9.iloc[j,1]], mode='auto') 
        L9_KS_statistic.append(statistic), L9_KS_pvalue.append(pvalue)
        j += 1

    print('K-S calculated statistics (compare to K-S test statistic):', np.asarray(L9_KS_statistic))
    print('K-S calculated p-values:', np.asarray(L9_KS_pvalue))
    
    KS_results = [0 for i in range(len(perm_9))]  #
    for k in range(len(perm_9)):   
        if L9_KS_statistic[k] > D_ks:   # if the calculated K-S statistic is > D_ks (the test statistic), then reject the null hypothesis that samples are drawn from the same distribution.
            KS_results[k] = 1  #  Value of 1 when the samples are from different distributions.
        # else:
        #     KS_results[k] = 2 
    #     print("Reject the null hyposthesis. The training and validation datasets are different at alpha =", alpha)
    # else:
    #     print("The training and validation datasets are the same")
    
    indices = [i for i, x in enumerate(KS_results) if x ==0]
    print('Experiment combinations that are not significantly different', str(perm_9.iloc[indices[0]]+1 ))  #Only values from columns > left column 1 (0) are valid.
    print(" ")
    # print(perm_9.iloc[indices[1]])
    print('Complete list of all experiments,perm_9',perm_9)
    s=0
    # K_S score = (100*sum([s+i for i in KS_results])/len(KS_results))
    print('K-S score, experiment:',100*sum([s+i for i in KS_results])/len(KS_results))
    #Comb = perm_9.iloc[:,:2]
    KS_results_df = pd.DataFrame(KS_results)
##    x = [perm_9.iloc[:,:2], np.asarray(KS_results).T]
##    print('x',x)
    print('KS_results, 1=> experiment combination is different:', KS_results_df)
    Result_df = pd.concat([perm_9,KS_results_df], axis=1)
    Result_df.columns = ['Ea', 'Eb', 'Diff']
    print(Result_df.head)

    count=pd.DataFrame()
    print('length perm_9', len(perm_9))
    
    ##np.savetxt("C:/Users/mark_/mark_data/Output/V24/KS_result.csv", Result_df, delimiter=',')


    
##    print('KS_results, 1 => from different distribution',Result) 
    #A = permar.iloc[j,0:3]  # A is the data frame of the permuted mechanical values used in each sample.
        #print(T.loc[i])
            #mSX_test_T = pd.concat([mSX_test_T,A],axis=1,ignore_index=True)
            #i += 1
    # return  KS_results
    #mSX_test = mSX_test_T.T  # matrix of mechanical predictors (does not include influent flow rate)
    # print(mSX_test.shape)
    # plt.plot(mSX_test)    
    return
   

