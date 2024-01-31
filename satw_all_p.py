# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 15:10:13 2020

@author: mark_
"""

#### Use this file to call the defined functions to prepare the full data (csv files) and to preprocess for NN building #####
#### Use this file to import the NN models for hyperparameter testing
#### Use this file to import and use the NN models to run simulations
# 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import tensorflow
from tensorflow import keras
# import tensorflow as tf
# import datetime
# import random
import csv
import scipy
import itertools
import os
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split 
from math import sqrt
from sklearn.metrics import mean_squared_error
from scipy import stats, mean
##
import statsmodels
from statsmodels.stats.multitest import multipletests
##from statsmodels.stats.weightstats import CompareMeans.ttest_ind
#statsmodels.stats.weightstats.CompareMeans.ttest_ind

####A = np.arange(125)
##A = np.asarray([(1,2,3,4,5,6),(11,12,13,14,15,16)])
##print('Length A',len(A))
##print('Shape A',np.shape(A))
##print('A',A)
##Af = A.flatten()
##print('A, flattened',Af)
##print('Length, A flattened',len(Af))
##print('Shape A flattened',np.shape(Af))
##C = Af.reshape((-1,2), order='C')
##print('B,(-1,2) C order',C)
##F = Af.reshape((-1,2), order='F')
##print('B, (-1,2), F order',F)
##
##copies = np.tile(Af,3)
##print('copies',copies)
##
##print(x)

# from autoinit import AutoInit

# print(sys.path, end="")

# from tensorflow import keras
# # from keras import backend as K
# from keras.models import Sequential, Model
# from keras.layers import Dense, Input, Dropout, Activation, Conv1D, Flatten, Reshape, LSTM, MaxPooling1D, GlobalAveragePooling1D, concatenate, Concatenate
# from keras.optimizers import RMSprop, SGD, Adam
# from keras.initializers import Constant, RandomUniform, glorot_normal, glorot_uniform
# #from keras import losses
# #from keras import metrics
# from keras.losses import mse, mae, mape, binary_crossentropy, sparse_categorical_crossentropy, mean_squared_logarithmic_error
# from keras.metrics import mse, mae, mape, accuracy	# the loss function is used when training the model # the metrics are not used when training the model. Any loss function may also be used as a metric funtion.
# from keras.models import load_model
# from keras.utils import plot_model, to_categorical
# from keras.layers.merge import concatenate
# from keras.models import model_from_json

##### ###### USE THESE WHEN RUNNING ON THE PC   ####################
from satw_data_p import NN_QX_mockdata, NN_preprocess, norm, autocorr, satw_ks_stats, KS_simulationdata, NN_architecture, reg_T_CV, NN_simulationdata #, NN_LX_traindata

##### Fit a curve to the experimentally acquired data
CV_red_kjd = reg_T_CV()  # The response CV reduction derived from the "reference experiment" in kJ/day

##########from satw_mlp_v5 import poly_rep, AFBR_MLP_model, AFBR_lstm, NN_DC
from satw_nn_p import afbr_lstm, poly_rep, AFBR_MLP_model, NN_DC, AFBR_MLP_sim
####

#########NN_LX_traindata()  # Use to get the ranked CV reduction of the mock experiments 
####
y_data, X_data, CV_ref_kjd, IFR_ref_all, IFR_M1  = NN_QX_mockdata()  # CV_ref_kjd = mock data created using the same predictors as used for the 'reference experiment' 
####
##y_data = pd.read_csv(open("C:/Users/mark_/mark_data/Input/y_data.csv"), delimiter=',', decimal=',', header=None)
##X_data = pd.read_csv(open("C:/Users/mark_/mark_data/Input/X_data.csv"),delimiter=',', decimal=',', header=None)

####print('length CV_red_kjd', len(CV_red_kjd))
####print(CV_red_kjd)
####print('length CV_ref_kjd', len(CV_ref_kjd))
####print(CV_ref_kjd)
##y200 = np.array([0,200])
##x200 = np.array([0,200])
##plt.figure(figsize=(14.1,10))
##plt.plot(x200,y200, c='k')
##plt.scatter(CV_red_kjd, CV_ref_kjd)
##plt.title('CV reduction, Reference experiment vs mock experiment')
##plt.xlabel('CV_red_kjd - reference experiment')
##plt.ylabel('CV_ref_kjd - mock experiment')
##plt.xlim(0,200)
##plt.ylim(0,200)
###plt.savefig('C:/Users/mark_/Documents/02_Professional/03_Unil/SATW/MDPI_paper/complementary_information/figures/Fig_CVred_ref_mock.svg', format="svg")
##plt.show()

#########################   SAVE raw Mock data  #######################################
####np.savetxt("C:/Users/mark_/mark_data/Input/y_data.csv", y_data, delimiter=",", fmt="%10.5f")
####np.savetxt("C:/Users/mark_/mark_data/Input/X_data.csv", X_data, delimiter=",", fmt="%10.5f") 
######
######### # # ## Set y_data_mm to the correct transformation in satw_data line x
######### # # # X_data, y_data, X_data_mm, y_data_mm, y_data_t, y_data_1p, y_data_yj, l9_CVred, CV_ref_all  = NN_LX_traindata()  # , ns, CV_ref_all, IFR_ref_all ns = number of samples. IFR = Influent Flow Rate
######### # # # np.savetxt("C:/Users/mark_/userdata/Output/X_data.csv", X_data_mm, delimiter=',')
######### # # # np.savetxt("C:/Users/mark_/userdata/Output/y_data.csv", y_data_mm, delimiter=',')
######### # # # np.savetxt("C:/Users/mark_/userdata/Output/X_data_mm.csv", X_data_mm, delimiter=',')
######### # # # np.savetxt("C:/Users/mark_/userdata/Output/y_data_mm.csv", y_data_mm, delimiter=',')
######### # # # np.savetxt("C:/Users/mark_/userdata/Output/l9_CVred.csv", l9_CVred, delimiter=',')
########

#############  Preprocess the transformed and Min-Max scaled data  
X_train_mm, X_test_mm, y_train_mm, y_test_mm, X_data_mm, y_data_mm, y_data_r_mm,  y_data_z_mm, y_data_t_mm, y_data_1p_mm, y_data_yj_mm, y_data_z, y_data_t, y_data_1p, y_data_yj = NN_preprocess(X_data, y_data)  #X_data, y_data, , y_data_z , y_data_z, y_data_z_mm


########X_data_mm, y_data_mm= NN_preprocess(X_data, y_data)
######### # # # # # X_train_mm = X_train_mm.to_numpy(copy=False)
######### # # # # # X_test_mm = X_test_mm.to_numpy(copy=False)
######### # # # # # y_train_mm = y_train_mm.to_numpy(copy=False)
######### # # # # # y_test_mm = y_test_mm.to_numpy(copy=False)
########
########### # # # #########  Save SHUFFLED data    ############  See satw_data line 2282 to change shuffle between True and False
##np.savetxt("C:/Users/mark_/mark_data/Input/y_data_r_mm.csv", y_data_r_mm, delimiter=",", fmt="%10.5f")   # Min-Max scaled raw response data before transformation
##np.savetxt("C:/Users/mark_/mark_data/Output/V24/y_data_mm.csv", y_data_mm, delimiter=",", fmt="%10.5f")   # Min-Max scaled response data after transformation
####np.savetxt("C:/Users/mark_/mark_data/Output/V24/X_data_mm.csv", X_data_mm, delimiter=",", fmt="%10.5f") 
##np.savetxt("C:/Users/mark_/mark_data/Output/X_train_mm.csv", X_train_mm, delimiter=",", fmt="%10.5f", newline="\n") 
##np.savetxt("C:/Users/mark_/mark_data/Output/y_train_mm.csv", y_train_mm, delimiter=",", fmt="%10.5f", newline="\n") 
##np.savetxt("C:/Users/mark_/mark_data/Output/X_test_mm.csv", X_test_mm, delimiter=",", fmt="%10.5f", newline="\n") 
##np.savetxt("C:/Users/mark_/mark_data/Output/V24/y_test_mm.csv", y_test_mm, delimiter=",", fmt="%10.5f", newline="\n")  
####
## #########  Save UN-SHUFFLED data    ############
##np.savetxt("C:/Users/mark_/mark_data/Output/V24/X_train_mm_ns.csv", X_train_mm, delimiter=",", fmt="%10.4f") 
##np.savetxt("C:/Users/mark_/mark_data/Output/V24/y_train_mm_ns.csv", y_train_mm, delimiter=",", fmt="%10.4f") 
##np.savetxt("C:/Users/mark_/mark_data/Output/V24/X_test_mm_ns.csv", X_test_mm, delimiter=",", fmt="%10.4f") 
##np.savetxt("C:/Users/mark_/mark_data/Output/V24/y_test_mm_ns.csv", y_test_mm, delimiter=",", fmt="%10.4f")

################## Load data for model building  ###########
####y_data = pd.read_csv(open("C:/Users/mark_/mark_data/Input/y_data.csv"), delimiter=',', decimal=',', header=None)
####X_data = pd.read_csv(open("C:/Users/mark_/mark_data/Input/X_data.csv"),delimiter=',', decimal=',', header=None)
##y_data_mm = pd.read_csv('C:/Users/mark_/mark_data/Output/V24/y_data_mm.csv', delimiter=',', header=None, engine='python')
##y_data_mm = pd.read_csv('C:/Users/mark_/mark_data/Input/y_data_mm.csv', delimiter=',', decimal='.', header=None, engine='python')
##X_data_mm = pd.read_csv("C:/Users/mark_/mark_data/Input/X_data_mm.csv",delimiter=',', decimal=',', header=None)
##X_train_mm = pd.read_csv("C:/Users/mark_/mark_data/Input/X_train_mm.csv",delimiter=',', decimal=',', header=None)
##y_train_mm = pd.read_csv("C:/Users/mark_/mark_data/Input/y_train_mm.csv",delimiter=',', decimal=',', header=None)
##X_test_mm = pd.read_csv("C:/Users/mark_/mark_data/Input/X_test_mm.csv",delimiter=',', decimal=',', header=None)
##y_test_mm = pd.read_csv("C:/Users/mark_/mark_data/Input/y_test_mm.csv",delimiter=',', decimal=',', header=None)

####### Pre-shuffled
##X_train_mm = np.loadtxt("C:/Users/mark_/mark_data/Input/X_train_mm.csv", delimiter=',')#, decimal=','   header=None
##y_train_mm = np.genfromtxt("C:/Users/mark_/mark_data/Input/y_train_mm.csv",delimiter=',')#, decimal=',', header=None)
##X_test_mm = np.genfromtxt("C:/Users/mark_/mark_data/Input/X_test_mm.csv",delimiter=',')#, decimal=',')#, header=None)
##y_test_mm = np.genfromtxt("C:/Users/mark_/mark_data/Input/y_test_mm.csv",delimiter=',')#, decimal=',')#, header=None)
##print('X_train_mm', X_train_mm)
##print('X_train_mm dtype', type(X_train_mm))
##print('X_train_mm', X_train_mm)
##print('X_test_mm', X_test_mm)
##print('y_test_mm', y_test_mm)

####### Not-shuffled
##X_train_mm = np.loadtxt("C:/Users/mark_/mark_data/Input/X_train_mm_ns.csv", delimiter=',')#, decimal=','   header=None
##y_train_mm = np.genfromtxt("C:/Users/mark_/mark_data/Input/y_train_mm_ns.csv",delimiter=',')#, decimal=',', header=None)
##X_test_mm = np.genfromtxt("C:/Users/mark_/mark_data/Input/X_test_mm_ns.csv",delimiter=',')#, decimal=',')#, header=None)
##y_test_mm = np.genfromtxt("C:/Users/mark_/mark_data/Input/y_test_mm_ns.csv",delimiter=',')#, decimal=',')#, header=None)
####print('X_train_mm', X_train_mm)
##print('X_train_mm dtype', type(X_train_mm))
##print('X_train_mm', X_train_mm)
##print('X_test_mm', X_test_mm)
##print('y_test_mm', y_test_mm)


##
##X_train_mm = X_train_mm.to_numpy()
##y_train_mm = y_train_mm.to_numpy()
##X_test_mm = X_test_mm.to_numpy()
##y_test_mm = y_test_mm.to_numpy()
##print('y_test_mm type, loaded',type(y_test_mm))

###########   Descriptive statistics and transformations to normalize raw data
##norm(y_data, y_data_z, y_data_t, y_data_1p, y_data_yj, y_data_mm, y_data_z_mm, y_data_t_mm, y_data_1p_mm, y_data_yj_mm)
########
############# K-S test to see if the training and testing datasets were drawn from the same distribution
##satw_ks_stats(y_train_mm, y_test_mm)
########
#############  Test of autocorrelation. Determination of VIF and R2 of the polynomial predictors 
##autocorr(CV_ref_kjd, X_data)


####################
##NN_architecture()   #  Evaluate effect of the number of MLP nodes, layers and the sample size, load archoptitest   #######
####################

##############################################################################################
##########  Build and evaluate the POLYNOMIAL MODEL. ns = not shuffled. _m = mean value
###### Returns the result of x replicates from poly_rep using loo method.
######  Load previously preprocessed shuffled or unshuffled data. Save Output.
##nrep=100  # number of replicate runs. Sets for all 3 models
##Yhat_v_poly = poly_rep(X_train_mm, y_train_mm, X_test_mm, y_test_mm, nrep)  #y_test_mm, rms_poly_mean, Rsq_mean,Wr_all_mean, Ir_all_meany_test_mm_ns,Yhat_polynomial_m,  y_test_mm, Poly_sum_df, rmse_poly, R2_poly
##
####test=[]
####for i in range(nrep):
####    test.append(Yhat_v_poly[i][:])
####Yhat_v_poly = np.asarray(Yhat_v_poly)
##Yhat_v_poly = np.ravel(Yhat_v_poly)
####print('Yhat_v_poly, ravel', Yhat_v_poly)
##print('Yhat_v_poly type', type(Yhat_v_poly))
##print('Yhat_v_poly shape', Yhat_v_poly.shape)
######Yhat_v_poly = Yhat_v_poly.reshape(-1,nrep, order='F')
######print('test', test)   
######print('Yhat_v_poly, ravel+reshape', Yhat_v_poly)
######print('Yhat_v_poly type', type(Yhat_v_poly))
######print('Yhat_v_poly shape', Yhat_v_poly.shape)
######Yhat_v_poly = Yhat_v_poly.astype(float)
###y_test_mm_loo = y_test_loo_all.reshape(-1, nrep, order='F')
##Yhat_v_poly = Yhat_v_poly.reshape(-1,nrep, order='F')
##np.savetxt("C:/Users/mark_/mark_data/Output/Yhat_v_poly.csv", Yhat_v_poly, delimiter=",", fmt='%s', newline='\n') # shuffled data , fmt="%10.4f"
##np.savetxt("C:/Users/mark_/mark_data/Output/y_test_mm.csv", y_test_mm, delimiter=",", fmt="%10.4f") # shuffled data , fmt="%10.4f"
####np.savetxt("C:/Users/mark_/mark_data/Output/V24/Yhat_v_poly_ns.csv", Yhat_v_poly, delimiter=",", fmt="%10.4f") ## non-shuffled data
####np.savetxt("C:/Users/mark_/mark_data/Output/V24/y_test_mm_ns.csv", y_test_mm, delimiter=",", fmt="%10.4f") # shuffled data , fmt="%10.4f"
####
####
######Yhat_v_poly = pd.read_csv("C:/Users/mark_/mark_data/Output/Yhat_v_poly_ravel.csv", engine='python', sep=',', header = 0, index_col=False, decimal=',' )
####Yhat_v_poly = np.genfromtxt(open("C:/Users/mark_/mark_data/Output/V24/Yhat_v_poly.csv"), delimiter=',')  #mlp_df.loc[:,'mlp_s']
#####np.genfromtxt
######print('test',test.values)
######test_2=[]
######for n in (len(test)):
######    test_2.append(test[:,n])
######print('test',np.asarray(test_2))
####
##### Analyse results of replicate polynomial runs.
##print('Yhat_v_poly shape',Yhat_v_poly.shape)  # len(y_test_mm) x nrep = 306x100
##poly_sl = []
##poly_pval=[]
##poly_rmse = []
##poly_reg = []
##poly_r2_score=[]
####
####poly_nobs, poly_minmax, poly_mean, poly_variance, _s, _k = scipy.stats.describe(Yhat_v_poly[:], axis=1, ddof=1, bias=True, nan_policy='propagate') # axis = 1 implies over each replicated day
####
####columns = ['Min', 'Max', 'Mean', 'Variance']#, 'Slope', 'p-val', 'RMSE', 'R2'],'Skewness','Kurtosis'
####len(columns)
####poly_sum = np.concatenate((poly_minmax[0], poly_minmax[1], poly_mean, poly_variance), axis=0).reshape(-1,len(columns), order='F')  #, , 
####poly_sum_df = pd.DataFrame(data = poly_sum)
####poly_sum_df.columns = columns
######poly_sum_df.to_csv("C:/Users/mark_/mark_data/Output/poly_sum_df.csv", sep=',', float_format='%.7f', header=True, decimal=',')
####
##print('y_test_mm',y_test_mm)
##print('Yhat_v_poly[:,2]',Yhat_v_poly[:,2])
##
##for n in range(nrep):
##    # result.append(scipy.stats.linregress(y_test_mm, yhat_poly))  #y_test_mm[:], Yhat_polynomial[:,j]
##        # res.append(scipy.stats.linregress(y_test_mm, Yhat_poly_all.T[n]))
##        # inter.append(scipy.stats.linregress(y_test_mm[:,0], Yhat_poly_all[:,n]).intercept) # Y intercept of the polynomial
##        # Rsq.append(scipy.stats.linregress(y_test_mm[:,0], Yhat_poly_all[:,n]).rvalue) #.rvalue
##    poly_sl.append(scipy.stats.linregress(y_test_mm[:,0], Yhat_v_poly[:,n]).slope) # slope of the regression line.reshape(-1,1).reshape(-1,1)
##    poly_pval.append(scipy.stats.linregress(y_test_mm[:,0], Yhat_v_poly[:,n]).pvalue)#
##    poly_reg.append(sklearn.linear_model.LinearRegression().fit(y_test_mm.reshape(1, -1), Yhat_v_poly[:,n].reshape(1, -1)))#
##    poly_rmse.append(sqrt(mean_squared_error(y_test_mm[:,0], Yhat_v_poly[:,n])))#
##    poly_r2_score.append(sklearn.metrics.r2_score(y_test_mm[:,0], Yhat_v_poly[:,n]))#.reshape(-1,1),Yhat_poly_all.T[n].reshape(-1,1))) #, fit_intercept=True
##print('poly scores, all reps',poly_sl)
##print('poly scores mean, all reps',np.mean(np.asarray(poly_sl)))
##print(np.asarray(poly_sl).shape)
####np.savetxt("C:/Users/mark_/mark_data/Output/V24/poly_sl_ns.csv", np.asarray(poly_sl), delimiter=",", fmt="%10.4f")
##print('poly pvals, all reps',poly_pval)
##print(np.asarray(poly_pval).shape)
##print(np.asarray(poly_rmse).shape)
##print('Poly_rmse mean', np.mean(np.asarray(poly_rmse)))
####np.savetxt("C:/Users/mark_/mark_data/Output/V24/poly_rmse_ns.csv", np.asarray(poly_rmse), delimiter=",", fmt="%10.4f")
##print(np.asarray(poly_r2_score).shape)
##print('Poly_r2_score mean', np.mean(np.asarray(poly_r2_score)))
##
##poly_skewness = scipy.stats.skew(Yhat_v_poly, axis=0, bias=True, nan_policy='propagate')
##poly_kurtosis = scipy.stats.kurtosis(Yhat_v_poly, axis=0,fisher=True, bias=True, nan_policy='propagate') # Fischer = True => normal = 0.0
##
####columns_corr_poly = ['poly_s', 'Poly_pval', 'poly_rmse', 'Poly_r2_score','Skewness','Kurtosis']
####poly_corr = np.concatenate((Poly_sl, Poly_pval, Poly_rmse, Poly_r2_score, poly_skewness , poly_kurtosis), axis=0).reshape(-1,len(columns_corr_poly), order='F')  #.reshape(-1,len(columns), order='F') axis = 0 => number of replicates
####poly_corr_df = pd.DataFrame(data = poly_corr)
####poly_corr_df.columns = columns_corr_poly
#####poly_corr_df.to_csv("C:/Users/mark_/mark_data/Output/poly_corr_df.csv", sep=',', float_format='%.4f', header=True, decimal=',')
## 
##print('Polynomial-Mean RMSE:', np.mean(poly_rmse))
##print('Polynomial-Mean R2:', np.mean(poly_r2_score))
##
#### Multiple pairwise comparison of p-values. Null hypothesis is no difference in p-values.
###  Returns "True" when the Null hypothesis of No difference (they are the same) can be rejected for the given alpha. The null hypothesis is that the observed difference is due to chance alone. 
##    
###print('Bonferroni - polynomial p-val',statsmodels.stats.multitest.multipletests(poly_corr_df.loc[:,'Poly_pval'], alpha=0.05, method='bonferroni', is_sorted=False, returnsorted=True))
### returns 1) reject null hypothesis?, 2) pvals_corrected, 3) alphacBonf (corrected alpha for Bonferroni method)
##
####Multiple pairwise comparison of p-values. Null hypothesis is no difference in p-values.
####Returns "True" when the Null hypothesis of No difference (they are the same) can be rejected for the given alpha. The null hypothesis is that the observed difference is due to chance alone. 
##    
####poly_nobs, poly_minmax, poly_mean, poly_variance, poly_skewness, poly_kurtosis = scipy.stats.describe(Poly_sum_df, axis=1, ddof=1, bias=True, nan_policy='propagate')
####columns = ['Min', 'Max', 'Mean', 'Variance','Skewness','Kurtosis']  #, 'p-val''No. samples', , 'Slope_R2', 'RMSE'
####len(columns)
######print(poly_minmax[0])
######print(poly_minmax[1])
######print(poly_nobs)
####poly_sum = np.concatenate((poly_minmax[0], poly_minmax[1], poly_mean, poly_variance, poly_skewness, poly_kurtosis), axis=0).reshape(-1,len(columns), order='F')   #poly_nobs, R2_poly, rmse_poly), axis=0).reshape(-1,len(columns), order='F')   #, poly_pval
####poly_sum_df = pd.DataFrame(data = poly_sum)
####poly_sum_df.columns = columns
######np.savetxt("C:/Users/mark_/mark_data/Output/Poly_sum_df.csv",  Poly_sum_df, delimiter=',')
####poly_sum_df.to_csv("C:/Users/mark_/mark_data/Output/poly_sum_df.csv", sep=',', float_format='%.4f', header=True, decimal=',')
##
##xl = np.array([0,1])
##yl = np.array([0,1]) 
##plt.xlim(0, 1)
##for i in range(nrep):
##    plt.scatter(y_test_mm, Yhat_v_poly[:,i], s=1)
##plt.title('Polynomial model - Predicted versus True values, testing ', fontsize='large') #MLP predicted and observed flow rate during testing
###plt.legend(labels)
##plt.plot(xl,yl,  c="k")
###plt.savefig('C:/Users/mark_/Documents/02_Professional/03_Unil/SATW/MDPI_paper/complementary_information/figures/Fig_polynomial_cross_sh.svg', format="svg")
##plt.show() 

##plt.figure(figsize=(16,6)) 
###labels=["True values", "Predictions"]
##plt.plot(y_test_mm, label = "True values" )
##plt.plot(poly_mean, label = "Predictions")
####for j in range(nrep):
####    plt.plot(Yhat_poly_all[:,j], label = "Predictions")
#####plt.plot(Yhat_polynomial_m)
##plt.title('Polynomial model - Mean predicted and True values, testing ', fontsize='large') #MLP predicted and observed flow rate during testing
##plt.legend() #labels
###plt.savefig('C:/Users/mark_/Documents/02_Professional/03_Unil/SATW/MDPI_paper/complementary_information/figures/Fig_polynomial_time_sh.svg', format="svg")
##plt.show()

#####  Polynomial model archives
##print("Polynomial rms error is: ", rms_poly_mean) #+ str(rms_poly_mean)  # Save, but do not use this line.
##print("Polynomial model validation test rms error is: ", rmse_poly) #+ str(rms_poly_mean)
##    
##print("Polynomial model validation test R2 is:" + str(Rsq_mean))
##print("Polynomial model validation test R2 is:", R2_poly)
##
## # np.savetxt("C:/Users/mark_/mark_data/Output/Poly_sum_df.csv", Poly_sum_df, delimiter=',')  # Poly_sum_df is?
##print("Polynomial model replication statistics:", Poly_sum_df)

#poly_nobs, poly_minmax, poly_mean, poly_variance, poly_skewness, poly_kurtosis = scipy.stats.describe(Yhat_poly_all, axis=0, ddof=1, bias=True, nan_policy='propagate') # axis = 1 implies over each replicated day

##print(poly_minmax, poly_mean, poly_variance, poly_skewness, poly_kurtosis
##print(poly_minmax[0].shape)
##print(poly_minmax[1].shape)
##print(poly_mean.shape)
##print(poly_variance.shape)
##print(poly_skewness.shape)
##print(poly_kurtosis.shape)
   
###### Chi-square test of the null hypothesis that there is no difference between the p-value or the slope of the linregress results.
####    
####chisq, p = scipy.stats.chisquare(poly_sum_df.loc[:,'p-val'])  #, Poly_sum_df.loc[:,'Slope']
####    
###### t-test of the null hypothesis that the difference in the means of 2 group means is zero.
####tstat, pval_t = scipy.stats.ttest_rel(poly_sum_df.loc[:,'p-val'], poly_sum_df.loc[:,'Slope'], axis=0, nan_policy='omit', alternative='two-sided')



###################################################################################################
##########  Build and evaluate the MLP MODEL

####A single run
##Yhat, Yhat_v_mlp,  rmse_mlp, result  = AFBR_MLP_model(X_train_mm, y_train_mm, X_test_mm, y_test_mm)  #Yhat, Yhat_v_mlp, Yhat_v_mlp_1_layer, rmse_mlp, r_value 
##np.savetxt("C:/Users/mark_/mark_data/Output/V24/Yhat_v_mlp.csv", Yhat_v_mlp, delimiter=",", fmt="%10.4f") # shuffled data
####np.savetxt("C:/Users/mark_/mark_data/Output/V24/Yhat_v_mlp_ns.csv", Yhat_v_mlp, delimiter=",", fmt="%10.4f") # non-shuffled data
##
##
##
###print("MLP model validation test rms error is ??? " , rmse_mlp)
##
##R2_mlp=result.rvalue  #The Pearson correlation coefficient. The square of rvalue is equal to the coefficient of determination.
##
##rms_mlp = sqrt(mean_squared_error(y_test_mm, Yhat_v_mlp))
##print("MLP model validation test rms error is: %.3f" % rms_mlp)  #+ str(rms_mlp)
##
##print("MLP model validation test r_value is: %.3f" , R2_mlp) # R2 = r_value squared 
##    
##reg_mlp = LinearRegression().fit(y_test_mm.reshape(-1,1),Yhat_v_mlp.reshape(-1,1)) #, fit_intercept=True
##print("MLP model validation test R2 is:" + str(reg_mlp.score(y_test_mm.reshape(-1,1),Yhat_v_mlp.reshape(-1,1))))  
####reg_mlp = LinearRegression().fit(y_test_mm[:,0],Yhat_v_m[:,0]) #, fit_intercept=True
####print("MLP model validation test R2 is:" + str(reg_mlp.score(y_test_mm[:,0],Yhat_v_m[:,0]))) 
####reg_mlp = LinearRegression().fit(y_test_mm,Yhat_v_m) #, fit_intercept=True
####print("MLP model validation test R2 is:" + str(reg_mlp.score(y_test_mm,Yhat_v_m))) 
##
##plt.figure()
##plt.plot(y_test_mm)
##plt.plot(Yhat_v_mlp)
##plt.legend()
##plt.title('MLP train and test') 
##plt.show()

####### Replicate MLP runs. Run on the HPC clusters or PC.

##Yhat_mlp_rep = []
##Yhat_v_mlp_all = []
##########mlp_rmse_rep = []
##########mlp_r_value_rep = []
##########
##
####mlp_rmse = []
####mlp_r2_score = []
##########
##nrep=2#nrep
##
##for i in range(nrep):
##     # pass
##    Yhat, Yhat_v_mlp, rmse_mlp = AFBR_MLP_model(X_train_mm, y_train_mm, X_test_mm, y_test_mm)# mom[i]satw_nn_p,inputs, outputs, inputs_test, outputs_test  result
##    Yhat_mlp_rep.append(Yhat)
##    Yhat_v_mlp_all.append(Yhat_v_mlp[:,0])#[:,0
####    mlp_rmse_rep.append(rmse_mlp)
####    mlp_r_value_rep.append(r_value)
###np.savetxt("C:/Users/mark_/mark_data/Output/V24/Yhat_v_mlp_ns.csv", np.asarray(Yhat_v_mlp_all).T, delimiter=",", fmt="%10.4f") # non-shuffled data
##########print('Yhat',Yhat)
####print('Yhat_v_mlp [:,0]',Yhat_v_mlp[:,0])
####print('Yhat_v_mlp sliced',Yhat_v_mlp[:3])
####print('y_test_mm[:,0]',y_test_mm[:,0]) #
####
####    
#####Yhat_v_mlp_rep_f = [item for sublist in Yhat_v_mlp_rep]
####print('orig Yhat_v_mlp_rep',Yhat_v_mlp_all)
#####print('flat',Yhat_v_mlp_rep_f)
##print('y_test_mm brute',y_test_mm)
####print('y_test_mm shape',y_test_mm.shape) # 306,1
####
####print('Yhat_v_mlp_all brute array',np.asarray(Yhat_v_mlp_all))
####print('Yhat_v_mlp_all flattened array',np.asarray(Yhat_v_mlp_all).flatten())
####print('recieved Yhat_v_mlp_all shape',np.asarray(Yhat_v_mlp_all).shape) 
##Yhat_v_mlp_all_t = np.asarray(Yhat_v_mlp_all).flatten()
##print('Yhat_v_mlp_all_t shape',Yhat_v_mlp_all_t.shape)   # 306,2
##Yhat_v_mlp_all_u = Yhat_v_mlp_all_t.reshape(len(y_test_mm), nrep, order='F')
##print('Yhat_v_mlp_u shape',Yhat_v_mlp_all_u.shape)   # 306,2
##print('Yhat_v_mlp_all transformed array',Yhat_v_mlp_all_u)
##Yhat_v_mlp_all = Yhat_v_mlp_all_u 
###mlp_mean_2 = np.mean(Yhat_v_mlp_all[0], axis = 1)
##
##mlp_nobs, mlp_minmax, mlp_mean, mlp_variance, _s, _k = scipy.stats.describe(Yhat_v_mlp_all[:,:], axis=1, ddof=1, bias=True, nan_policy='propagate')
##columns = ['y_test_mm','Min', 'Max', 'Mean', 'Variance']#, 'Slope', 'p-val', 'RMSE', 'R2'],'Skewness','Kurtosis'
##len(columns)
##
####print('y_test_mm[:,0] shape',y_test_mm[:,0].shape)
####print('mlp_minmax[0] shape',mlp_minmax[0].shape)
####print('mlp_minmax[1] shape',mlp_minmax[1].shape)
####print('mlp_mean shape',mlp_mean.shape)
####print('mlp_variance shape',mlp_variance.shape)
####
####print('mlp_mean',mlp_mean)
####print('y_test_mm[:,0]',y_test_mm[:,0])
####mlp_r2_score_s.append(sklearn.metrics.r2_score(y_test_mm[:,0], mlp_mean))
####print('mlp_r2_score_separate', mlp_r2_score_s)
##
##mlp_sum = np.concatenate((y_test_mm[:,0] ,mlp_minmax[0], mlp_minmax[1],  mlp_mean, mlp_variance), axis=0).reshape(-1,len(columns), order='F')  #, , 
##mlp_sum_df = pd.DataFrame(data = mlp_sum)
##mlp_sum_df.columns = columns
###mlp_sum_df.to_csv("C:/Users/mark_/mark_data/Output/mlp_sum_df.csv", sep=',', float_format='%.7f', header=True, decimal=',')
##
###Yhat_v_mlp = np.genfromtxt(open("C:/Users/mark_/mark_data/Output/V24/Yhat_v_mlp_ns.csv"), delimiter=',')
######
######
######print('Yhat_v_mlp[:,1]', Yhat_v_mlp[:,1])
######print('y_test_mm[:]', y_test_mm[:])
######
##mlp_sl = []
##mlp_pval=[]
##mlp_rmse = []
##mlp_r2_score =[]
##
##for n in range(nrep):
##    # result.append(scipy.stats.linregress(y_test_mm, yhat_poly))  #y_test_mm[:], Yhat_polynomial[:,j]
##        # res.append(scipy.stats.linregress(y_test_mm, Yhat_poly_all.T[n]))
##        # inter.append(scipy.stats.linregress(y_test_mm[:,0], Yhat_poly_all[:,n]).intercept) # Y intercept of the polynomial
##        # Rsq.append(scipy.stats.linregress(y_test_mm[:,0], Yhat_poly_all[:,n]).rvalue) #.rvalue
##    mlp_sl.append(scipy.stats.linregress(y_test_mm[:], Yhat_v_mlp[:,n]).slope) # slope of the regression line.reshape(-1,1).reshape(-1,1)
##    mlp_pval.append(scipy.stats.linregress(y_test_mm[:], Yhat_v_mlp[:,n]).pvalue)#
##        # Poly_reg.append(linear_model.LinearRegression().fit(y_test_mm[:,0], Yhat_poly_all[:,n]))#
##    mlp_rmse.append(sqrt(mean_squared_error(y_test_mm[:], Yhat_v_mlp[:,n])))#
##    mlp_r2_score.append(sklearn.metrics.r2_score(y_test_mm[:],Yhat_v_mlp[:,n]))#.reshape(-1,1),Yhat_poly_all.T[n].reshape(-1,1))) #, fit_intercept=True   mlp_mean
##print('MLP slope shape',np.asarray(mlp_sl).shape)
##print('MLP slope',mlp_sl)
##print('MLP mean slope',np.mean(np.asarray(mlp_sl)))
###np.savetxt("C:/Users/mark_/mark_data/Output/V24/mlp_sl_ns.csv", np.asarray(mlp_sl), delimiter=",", fmt="%10.4f")
##print('MLP pvalue shape',np.asarray(mlp_pval).shape)
##print('MLP pval', mlp_pval)
##print('MLP rmse shape',np.asarray(mlp_rmse).shape)
##print('MLP rmse',mlp_rmse)
##print('MLP mean rmse',np.mean(np.asarray(mlp_rmse)))
###np.savetxt("C:/Users/mark_/mark_data/Output/V24/mlp_rmse_ns.csv", np.asarray(mlp_rmse), delimiter=",", fmt="%10.4f")
##print('MLP r2_score shape',np.asarray(mlp_r2_score).shape)
##print('MLP r2_score', mlp_r2_score)
##print('MLP r2_score', np.mean(np.asarray(mlp_r2_score)))
##
####
##mlp_skewness = scipy.stats.skew(Yhat_v_mlp_all, axis=0, bias=True, nan_policy='propagate')
##mlp_kurtosis = scipy.stats.kurtosis(Yhat_v_mlp_all, axis=0,fisher=True, bias=True, nan_policy='propagate') # Fischer = True => normal = 0.0
##
##columns_corr_mlp = ['mlp_s', 'MLP_pval', 'mlp_rmse', 'MLP_r2_score','Skewness','Kurtosis']
##mlp_corr = np.concatenate((mlp_sl, mlp_pval, mlp_rmse, mlp_r2_score, mlp_skewness , mlp_kurtosis), axis=0).reshape(-1,len(columns_corr_mlp), order='F')  #.reshape(-1,len(columns), order='F') axis = 0 => number of replicates
##mlp_corr_df = pd.DataFrame(data = mlp_corr)
##mlp_corr_df.columns = columns_corr_mlp
###mlp_corr_df.to_csv("C:/Users/mark_/mark_data/Output/mlp_corr_df.csv", sep=',', float_format='%.4f', header=True, decimal=',')
##
##print('MLP-Mean RMSE:', np.mean(mlp_rmse))
##print('MLP-Mean R2:', np.mean(mlp_r2_score))
#### Multiple pairwise comparison of p-values. Null hypothesis is no difference in p-values.
###  Returns "True" when the Null hypothesis of No difference (they are the same) can be rejected for the given alpha. The null hypothesis is that the observed difference is due to chance alone. 
##    
##print('Bonferroni - MLP',statsmodels.stats.multitest.multipletests(mlp_corr_df.loc[:,'MLP_pval'], alpha=0.05, method='bonferroni', is_sorted=False, returnsorted=True))
### returns 1) reject null hypothesis?, 2) pvals_corrected, 3) alphacBonf (corrected alpha for Bonferroni method)


############################
######  Compare Poly - MLP t-test

####Yhat_v_poly_t = pd.read_csv(open("C:/Users/mark_/mark_data/Output/V24/Yhat_v_poly.csv"), delimiter=',', decimal=',', header=None)
####Yhat_v_mlp_t = pd.read_csv(open("C:/Users/mark_/mark_data/Output/V24/Yhat_v_mlp.csv"),delimiter=',', decimal=',', header=None)
####
####print(type(Yhat_v_poly_t))
####print(type(Yhat_v_mlp_t))

##Yhat_v_poly_t = np.genfromtxt("C:/Users/mark_/mark_data/Output/V24/Yhat_v_poly.csv", delimiter=',')
##Yhat_v_mlp_t = np.genfromtxt("C:/Users/mark_/mark_data/Output/V24/Yhat_v_mlp.csv",delimiter=',')
##
##print(type(Yhat_v_poly_t))
##print('Yhat_v_poly_t shape', Yhat_v_poly_t.shape)
##print('Yhat_v_poly',Yhat_v_poly_t)
##print(type(Yhat_v_mlp_t))
##print('Yhat_v_mlp_t shape',Yhat_v_mlp_t.shape)
##print('Yhat_v_mlp_t',Yhat_v_mlp_t)
##
##
###np.genfromtxt(open("C:/Users/mark_/mark_data/Output/poly_corr_df.csv"), delimiter=',', names=True, max_rows=20)
####np.savetxt("C:/Users/mark_/mark_data/Output/V24/Yhat_v_mlp_ns.csv", Yhat_v_mlp_all, delimiter=",", fmt="%10.4f") # non-shuffled data
####The p-value quantifies the probability of observing as or more extreme values assuming the null hypothesis, that the samples are drawn from populations with the same population means, is true. 
#### T-test null hypothesis: The 2 independent samples have identicle expected values. 
##t_test = scipy.stats.ttest_ind(Yhat_v_poly_t, Yhat_v_mlp_t, axis=0, equal_var=True, nan_policy='propagate', permutations=None, random_state=None, alternative='two-sided', trim=0)  #.to_numpy(dtype='float32')
##print('t-test statistic', t_test.statistic)
##print('t-test p-value', t_test.pvalue)  #The p-value quantifies the probability of observing as or more extreme values assuming the null hypothesis (same mean), that the samples are drawn from populations with the same population means, is true.
###print('t-test p-value', t_test[1])
##print('Bonferroni - polynomial - predictions. TRUE => reject the NULL hypothesis => The samples have DIFFERENT average (expected) values (they are not the same), reject H0.')
#####
##bonf = 0.01/len(np.asarray(Yhat_v_mlp_t).T) #Bonferroni correction for the 0.05 alpha level.
##print('Bonferoni correction value:', bonf)
##print('Poly-MLP. Number different not by chance, Bonferroni, alpha/m',len(t_test[1][np.where(t_test[1]>bonf)]))  # Number of samples where the difference is not due to chance (p-value > alpha (with Bonferoni correction)
##print('Poly-MLP. Number different not by chance, alpha = 0.01?',len(t_test[1][np.where(t_test[1]>0.01)]))

####The p-value quantifies the probability of observing as or more extreme values assuming the null hypothesis, that the samples are drawn from populations with the same population means, is true.
####A p-value larger than a chosen threshold (e.g. 5% or 1%) indicates that our observation is not so unlikely to have occurred by chance.
##plt.figure()
##plt.plot(t_test[0])  # t-test statistic
##plt.title('t-test statistic')
##plt.show()
##         
##plt.figure()
##plt.plot(t_test[1])  # t-test p-value
##plt.title('t-test p-value')
##plt.show()
##print('Bonferroni - polynomial - predictions. TRUE => reject the NULL hypothesis => The samples have DIFFERENT average (expected) values (they are not the same), reject H0.')
##print('Bonferroni - polynomial - predictions. t-statistic:, pvalue:',statsmodels.stats.multitest.multipletests(t_test[1], alpha=0.00000001, method='bonferroni', is_sorted=False, returnsorted=True))
##print('alpha level for poly-MLP t-test:', 0.05)
#returns 1) reject null hypothesis?, 2) pvals_corrected, 3) alphacBonf (corrected alpha for Bonferroni method)
##Returns "true" for hypothesis that can be rejected for given alpha. TRUE means that the samples ARE DIFFERENT.       
##Yhat_v_poly.to_csv("C:/Users/mark_/mark_data/Output/Yhat_v_poly_df.csv", sep=',', float_format='%.5f', header=True, decimal=',')




########################################################
########  Build and evaluate the LSTM model
#### Reshape for lstm nn

##### LSTM 
##Yhat, Yhat_v_lstm, inputs_test, outputs_test, scale_tanh = afbr_lstm(X_data, y_data) #inputs_train, outputs_train, inputs_test, outputs_test,  X_data_mm_tr, X_data_mm_ts, y_data_mm_tr, y_data_mm_ts()
####print(Yhat_v_lstm.shape)
####print(outputs_test.shape)
####print(Yhat_v_lstm)
####print(outputs_test)
##

##
##
##rms_lstm = sqrt(mean_squared_error(scale_tanh.inverse_transform(outputs_test.reshape(-1,1)),scale_tanh.inverse_transform(Yhat_v_lstm.reshape(-1,1))))
##print("LSTM validation test rms error is: %.3f " % rms_lstm)
##
##rms_lstm = sqrt(mean_squared_error(outputs_test.reshape(-1,1), Yhat_v_lstm.reshape(-1,1)))
##print("LSTM validation test rms error is: %.3f" % rms_lstm)
##    
##reg_lstm = LinearRegression().fit(outputs_test.reshape(-1,1),Yhat_v_lstm.reshape(-1,1)) #, fit_intercept=True
##print("LSTM model validation test R2: %.3f" % reg_lstm.score(outputs_test.reshape(-1, 1),Yhat_v_lstm.reshape(-1, 1)))   
##
##
##plt.figure()
##plt.plot(scale_tanh.inverse_transform(outputs_test.reshape(-1,1)))
##plt.plot(Yhat_v_lstm)
##plt.show()
##
########### Replicate LSTM runs
##n=0
##m=0
##nrep = 100#nrep #len(outputs_test)
##Yhat_v_lstm_all = []
##lstm_sl = []
##lstm_pval = []
##lstm_rmse = []
##lstm_r2_score = []
##
##for n in range(nrep):
##    Yhat, Yhat_v_lstm, inputs_test, outputs_test, scale_tanh = afbr_lstm(X_data, y_data)  #inputs_train, outputs_train, inputs_test, outputs_test
##    Yhat_v_lstm_all.append(np.asarray(Yhat_v_lstm[:,0])) 
##     # result.append(scipy.stats.linregress(y_test_mm, yhat_poly))  #y_test_mm[:], Yhat_polynomial[:,j]
##         # res.append(scipy.stats.linregress(y_test_mm, Yhat_poly_all.T[n]))
##         # inter.append(scipy.stats.linregress(y_test_mm[:,0], Yhat_poly_all[:,n]).intercept) # Y intercept of the polynomial
##         # Rsq.append(scipy.stats.linregress(y_test_mm[:,0], Yhat_poly_all[:,n]).rvalue) #.rvalue
##
##Yhat_v_lstm_all = np.asarray(Yhat_v_lstm_all).T
##print('Yhat_v_lstm_all shape',Yhat_v_lstm_all.shape)
####
####np.savetxt("C:/Users/mark_/mark_data/Output/V24/Yhat_v_lstm.csv", Yhat_v_lstm, delimiter=",", fmt="%10.4f") # shuffled data
####np.savetxt("C:/Users/mark_/mark_data/Output/V24/y_test_lstm.csv", outputs_test[:,0], delimiter=",", fmt="%10.4f") # shuffled data 
######np.savetxt("C:/Users/mark_/mark_data/Output/V24/Yhat_v_lstm_ns.csv", np.mean(Yhat_v_lstm_all, axis=1), delimiter=",", fmt="%10.4f") # non-shuffled data
######np.savetxt("C:/Users/mark_/mark_data/Output/V24/y_test_lstm_ns.csv", outputs_test[:,0], delimiter=",", fmt="%10.4f") # non-shuffled data
####    
####lstm_nobs, lstm_minmax, lstm_mean, lstm_variance, _s, _k = scipy.stats.describe(Yhat_v_lstm_all, axis=1, ddof=1, bias=True, nan_policy='propagate')
####columns = ['Min', 'Max', 'Mean', 'Variance']
####lstm_sum = np.concatenate((lstm_minmax[0], lstm_minmax[1], lstm_mean, lstm_variance), axis=0).reshape(-1,len(columns), order='F')   
####lstm_sum_df = pd.DataFrame(data = lstm_sum)
####lstm_sum_df.columns = columns
#####np.savetxt("C:/Users/mark_/mark_data/Output/lstm_sum_df.csv", lstm_sum_df, delimiter=',')
####lstm_sum_df.to_csv("C:/Users/mark_/mark_data/Output/lstm_sum_df.csv", sep=',', float_format='%.4f', header=True, decimal=',')
##
##for m in range(nrep):
##    lstm_sl.append(scipy.stats.linregress(outputs_test[:,0], Yhat_v_lstm_all[:,m]).slope) # slope of the regression line
##    lstm_pval.append(scipy.stats.linregress(outputs_test[:,0], Yhat_v_lstm_all[:,m]).pvalue)#
##    lstm_rmse.append(sqrt(mean_squared_error(outputs_test[:,0], Yhat_v_lstm_all[:,m])))
##    lstm_r2_score.append(sklearn.metrics.r2_score(outputs_test[:,0], Yhat_v_lstm_all[:,m]))
##np.savetxt("C:/Users/mark_/mark_data/Output/lstm_sl.csv", np.asarray(lstm_sl), delimiter=",", fmt="%10.4f")
##np.savetxt("C:/Users/mark_/mark_data/Output/lstm_rmse.csv", np.asarray(lstm_rmse), delimiter=",", fmt="%10.4f")
##lstm_skewness = scipy.stats.skew(Yhat_v_lstm_all, axis=0, bias=True, nan_policy='propagate')
##lstm_kurtosis = scipy.stats.kurtosis(Yhat_v_lstm_all, axis=0,fisher=True, bias=True, nan_policy='propagate') # Fischer = True => normal = 0.0
##
##columns_corr_lstm = ['lstm_s', 'p-val', 'lstm_rmse', 'Poly_r2_score','Skewness','Kurtosis'] 
##lstm_corr = np.concatenate((lstm_sl, lstm_pval, lstm_rmse, lstm_r2_score, lstm_skewness , lstm_kurtosis), axis=0).reshape(-1,len(columns_corr_lstm), order='F')  #.reshape(-1,len(columns), order='F') axis = 0 => number of replicates
##lstm_corr_df = pd.DataFrame(data = lstm_corr)
##lstm_corr_df.columns = columns_corr_lstm
##lstm_corr_df.to_csv("C:/Users/mark_/mark_data/Output/lstm_corr_df.csv", sep=',', float_format='%.4f', header=True, decimal=',')
##
##
##print('LSTM-Mean RMSE:', np.mean(lstm_rmse))
##print('LSTM-Mean coef of determ (R2):', np.mean(lstm_r2_score))

### Multiple pairwise comparison of p-values. Null hypothesis is no difference in p-values.
###  Returns "True" when the Null hypothesis of No difference (they are the same) can be rejected for the given alpha. The null hypothesis is that the observed difference is due to chance alone. 
##    
##print('Bonferroni - LSTM',statsmodels.stats.multitest.multipletests(lstm_corr_df.loc[:,'p-val'], alpha=0.05, method='bonferroni', is_sorted=False, returnsorted=True))
####    # returns 1) reject null hypothesis?, 2) pvals_corrected, 3) alphacBonf (corrected alpha for Bonferroni method)

# ##Save results of runs using shuffled data
# S_res = pd.DataFrame(np.column_stack((y_test_mm, Yhat_polynomial_m, Yhat_v_m[:,0], Yhat_v_l)))
# S_res.columns = ["y_test_mm", "Yhat_polynomial_m", "Yhat_v_m", "Yhat_v_l"]

# ##Save results of runs using shuffled data
# NS_res = pd.DataFrame(np.column_stack((y_test_mm, Yhat_polynomial_m, Yhat_v_m[:,0], Yhat_v_l)))
# NS_res.columns = ["y_test_mm_ns", "Yhat_polynomial_m_ns", "Yhat_v_m_ns", "Yhat_v_l_ns"]


#############  Evaluate the experimental plan ################
##############################################################
##
############# Use the KS statistic to compare the mock experiments and find those that are not significantly different.    
#KS_simulationdata(IFR_M1)  # Formerly l9_CVred
##
###print('y_data_mm shape',y_data_mm.shape) 
### ########  Build and evaluate the MLP model with columns removed from the datasets. dc = dropped columns
##y_data_mm = pd.DataFrame(y_data_mm)  ##Needed to make the subsequent code work
##X_data_mm = pd.DataFrame(X_data_mm)
####
##### #####  Manual delete
##### ##Delete the selected redundant experiment => 3 (E4) from predictors
##mdel = 3   # Manually delete a selected column from responses. Since experiments 3 and 9 are redundant, choose to remove experiment 3 (=> index 2)
##y_data_in = np.reshape(y_data_mm.to_numpy(dtype='float32'), (170,-1),  order='F') #
##y_data_mm_dc_man = np.delete(y_data_in, mdel, axis=1).reshape((-1,1), order = 'F')  # Delete the experiment from the array and then reshape to form a vector
##
##X_data_mm_dc_man = np.split(X_data_mm.to_numpy(dtype='float32'), 9) #X_data
##    
##ES1 = X_data_mm_dc_man[0], 
##ES2 = X_data_mm_dc_man[1], 
##ES3 = X_data_mm_dc_man[2], 
##ES4 = X_data_mm_dc_man[3], 
##ES5 = X_data_mm_dc_man[4], 
##ES6 = X_data_mm_dc_man[5], 
##ES7 = X_data_mm_dc_man[6], 
##ES8 = X_data_mm_dc_man[7], 
##ES9 = X_data_mm_dc_man[8] 
##
## # cols = ['ES1', 'ES2', 'ES3', 'ES4', 'ES5', 'ES6', 'ES7', 'ES8', 'ES9']
## # DataFrame.to_dict(orient='dict', into=<class 'dict'>)
##
##L9 = {0:ES1, 1:ES2, 2:ES3, 3:ES4, 4:ES5, 5:ES6, 6:ES7, 7:ES8, 8:ES9} 
##keys= np.arange(0,9)
##
##del X_data_mm_dc_man[mdel]  ##Delete the experiment from the array
## # cc= bb.pop(bb.index(mdel),axis=1)
##X_data_mm_dc_man = np.concatenate(X_data_mm_dc_man, axis=0)#.astype('float') # dc = dropped column. New array with selected experiment removed
##X_train_mm_dc_man, X_test_mm_dc_man, y_train_mm_dc_man, y_test_mm_dc_man = train_test_split(X_data_mm_dc_man, y_data_mm_dc_man, test_size=0.2, random_state= 42, shuffle = True, stratify= None)  #shuffle = True, random_state=42, stratify= None
####    Yhat, Yhat_v_mlp = AFBR_MLP_model(X_train_mm_dc, y_train_mm_dc, X_test_mm_dc, y_test_mm_dc) 
####X_train_mm, X_test_mm, y_train_mm, y_test_mm,  X_data_mm, y_data_mm, y_data_z_mm, y_data_t_mm, y_data_1p_mm, y_data_yj_mm, y_data_z, y_data_t, y_data_1p, y_data_yj
###X_train_mm, X_test_mm, y_train_mm, y_test_mm = NN_preprocess(X_data_mm_dc, y_data_mm_dc) 
##Yhat_man, Yhat_v_man,_ = AFBR_MLP_model(X_train_mm_dc_man, y_train_mm_dc_man, X_test_mm_dc_man, y_test_mm_dc_man)
###y_test_mm_dc_man = y_test_mm
##rms_dcm = sqrt(mean_squared_error(y_test_mm_dc_man,Yhat_v_man))
##print("MLP, manual delete validation test rms error is: " + str(rms_dcm))
##    
##reg_dcm = LinearRegression().fit(y_test_mm_dc_man.reshape(-1,1),Yhat_v_man.reshape(-1,1)) #, fit_intercept=True
##print("MLP, manual delete validation test R2: %.5f" % reg_dcm.score(y_test_mm_dc_man.reshape(-1,1),Yhat_v_man.reshape(-1,1)))
##
#######################
###### #### Random delete
##nrep = 10#nrep
##Y_train_rep = []
##Y_v_rep = []
##Yhat_rep = []
##Yhat_v_rep = []
##rms_dc_rep = []
##R2_dc_rep = []
##y_test_mm_dc_rep = []
##
##  # y_data, X_data  = NN_QX_mockdata()  # , y_data_t, y_data_1p, y_data_yj, l9_CVred, CV_ref_allNN_LX_traindata()
##  # X_train_mm, X_test_mm, y_train_mm, y_test_mm, y_data_t, y_data_1p, y_data_yj = NN_preprocess(X_data, y_data)
##
##for i in range(nrep):
##    Yhat_dc, Yhat_v_dc, y_test_mm_dc  = NN_DC(y_data_mm, X_data_mm) #DC = "Delete column". might need to add .to_numpy(dtype='float32'). rms_mlp_dc_mean, Rsq_dc_mean, Yhat_v_dc_mean, y_test_mm
##    #y_test_mm_dc  = NN_DC(y_data_mm, X_data_mm)
##
##    rms_dc = sqrt(mean_squared_error(y_test_mm_dc, Yhat_v_dc))
##    rms_dc_rep.append(rms_dc)
##
##    reg_dc_fit = LinearRegression().fit(y_test_mm_dc.reshape(-1, 1),Yhat_v_dc.reshape(-1, 1)) #, fit_intercept=True. y_test_mm_dc.reshape(-1,1),Yhat_v_dc.reshape(-1,1)
##    R2_dc = reg_dc_fit.score(y_test_mm_dc.reshape(-1, 1), Yhat_v_dc.reshape(-1, 1))#y_test_mm_dc.reshape(-1,1),Yhat_v_dc.reshape(-1,1)
##    R2_dc_rep.append(R2_dc)
##
##    Yhat_rep.append(Yhat_dc)
##    Yhat_v_rep.append(Yhat_v_dc)
##    
##    y_test_mm_dc_rep.append(np.asarray([np.array(j) for j in y_test_mm_dc]))
##print('Yhat_v_dc length', len(Yhat_v_dc))
##
##print('Yhat_v_rep length', len(Yhat_v_rep))
## 
##Yhat_train_all= np.concatenate([np.array(j) for j in Yhat_rep], axis = 1) #Concatenate the the training predictions. Convert list to numpy array. Yhat_rep # axis = 0 -> by rows
##Yhat_test_all_dc= np.concatenate([np.array(j) for j in Yhat_v_rep], axis = 1) #Concatenate the validation predictions. Convert list to numpy array.#Yhat_v_rep = predicted values from validation runs
###rms_all_dc = np.concatenate([np.array(j) for j in rms_dc_rep], axis = 1)
##print('Yhat_train_all length', len(Yhat_train_all)) 
##Yhat_v_dc_mean = np.mean(Yhat_test_all_dc, axis =1)
##
##y_test_mm_dc_reps = np.tile(y_test_mm_dc,nrep)
##        
######rms_dc = sqrt(mean_squared_error(y_test_mm_dc, Yhat_v_dc_mean))
####rms_dc = sqrt(mean_squared_error(y_test_mm_dc_rep, Yhat_test_all_dc))
####print("MLP, drop column validation test rms error is: " + str(rms_dc))
##
######reg_dc = LinearRegression().fit(y_test_mm_dc.reshape(-1,1),Yhat_v_dc_mean.reshape(-1,1)) #, fit_intercept=True   
####reg_dc = LinearRegression().fit(y_test_mm_dc_rep.reshape(-1,1),Yhat_test_all_dc.reshape(-1,1)) #, fit_intercept=True
####print("MLP model validation test R2 is:" + str(reg_dc.score(y_test_mm_dc_rep.reshape(-1,1),Yhat_test_all_dc.reshape(-1,1))))
##
##print("Mean MLP, random drop column validation test rms error is: " + str(np.asarray(rms_dc_rep).mean()))
##print("Mean MLP random drop column validation test R2 is:" + str(np.asarray(R2_dc_rep).mean()))

####reg_dc2 = LinearRegression().fit(y_test_mm_dc.reshape(-1, 1),Yhat_v_dc_mean.reshape(-1, 1)) #, fit_intercept=True[:,0]
####print("MLP model validation test R2 is:" + str(reg_dc2.score(y_test_mm_dc.reshape(-1, 1),Yhat_v_dc_mean.reshape(-1, 1))))   
##
## #### Plot regression of results using deleted columns
##yl = np.array([0,1,0.1])
##xl = np.array([0,1, 0.1])
##    
##fig, ax = plt.subplots()
##mk = 3
##ax.plot(xl,yl,  c="k")
## # ax.scatter(y_test_mm, Yhat_polynomial_m, s=mk, c='b', marker='v')# 4th degree polynomial
##ax.scatter(y_test_mm_dc, Yhat_v_dc, s=mk, c='r', marker='D', label='random delete') #  MLP
##ax.scatter(y_test_mm_dc_man, Yhat_v_man, s=mk, c='g', marker='s', label='manual delete, exp n°2')  # LSTM
## # ax.scatter(y_test_mm_t, Yhat_v, s=mk, c='k', marker='+')
## # ax.set_title("Shuffled data")
###plt.ylim(0, 1)
##ax.set_title("Predictions made with 1 experiment removed manually or randomly")
##ax.set(xlabel='True values', ylabel='Predictions')
##plt.legend()
###fig.savefig('C:/Users/mark_/Documents/02_Professional/03_Unil/SATW/MDPI_paper/complementary_information/figures/Fig_exp_plan_dc.svg', format="svg")  
##plt.show()
##

##################   Simulations using the MLP model  ##################################

###from satw_data import NN_simulationdata  # Run ths line only when not running replicate simulations
##perm_5 = list(itertools.product([4,8,12,24,36],[1,1.85,2.7,6.85,11],[0.5,1.15,1.8,2.9,4]))
##SX_test_mm, Q_Ex_sim = NN_simulationdata(perm_5)  # Run this line only when not runing replicate simulations

# ############################################################################################   

# # inputs_train, outputs_train, inputs_test, outputs_test = X_train_mm, y_train_mm, X_test_mm, y_test_mm
# # Yhat, Yhat_v_l, outputs_test_r = AFBR_lstm(inputs_train, outputs_train, inputs_test, outputs_test)

# # rms_lstm = sqrt(mean_squared_error(outputs_test_r,Yhat_v_l))
# # print("LSTM validation test rms error is: " + str(rms_lstm))
    
# # reg_lstm = LinearRegression().fit(outputs_test.reshape(-1,1),Yhat_v_l.reshape(-1,1)) #, fit_intercept=True
# # print("LSTM model validation test R2: %.2f" % reg_lstm.score(outputs_test_r.reshape(-1,1),Yhat_v_l.reshape(-1,1)))   



# # X_train_mm, y_train_mm, X_test_mm, y_test_mm = inputs.values, outputs.values, inputs_test.values, outputs_test.values 
# # X_train_mm, y_train_mm, X_test_mm, y_test_mm = tuple(list(inputs.itertuples(index=False, name=None))), outputs , list(inputs_test.itertuples(index=False, name=None)), outputs_test  #list(outputs.itertuples(index=False, name=None))list(outputs_test.itertuples(index=False, name=None)

# # # # plt.plot(x_data_mm)
# # # # # plt.plot(y_data_mm)
# # # # mean = np.mean(np.log1p(y_data))

# # # # # # mean_y = np.mean(y_data)
# # # # # # y_trans_exp = np.expm1((y_data + abs(y_data.min())) / 200)


# # # # # # from numpy.random import default_rng
# # # # # # # rng = np.random.RandomState(304)
# # # # # # y_gaussian = default_rng().standard_normal(len(y_data))
# # # # # # # loc = np.mean(y_data)
# # # # # # # y_gaussian = rng.normal(loc=loc, size=(len(y_data),1))
# # # # # # plt.hist(y_gaussian, bins=100)

# # # # # # plt.figure()
# # # # # # # plt.hist(y_trans_exp, bins=100)
# # # # # # plt.hist((np.divide(y_data,mean_y)), bins=100, alpha= 1, label="Raw response data, normalized")
# # # # # # plt.hist(y_trans_log, bins=100, alpha=0.75, label="log transformed response data")
# # # # # # plt.hist(y_gaussian, bins=100, alpha= 0.5, label="Normal distribution")
# # # # # # plt.legend()
# # # # # # plt.show()

# # # # # # plt.scatter(y_gaussian,y_data )
# # # # # # plt.scatter(y_gaussian,y_trans_log )


#############################################################################
#############################################################################
###    ARCHITECTURE DEVELOPMENT USE WHEN RUNNING ON THE CLUSTERS  OR PC #########
## Place the preprocessed data on the cluster: X_train_mm, y_train_mm, X_test_mm, y_test_mm
##For a single ANN model run
## Yhat, Yhat_v = AFBR_MLP_model(X_train_mm.to_numpy(dtype=float), y_train_mm, X_test_mm.to_numpy(dtype=float), y_test_mm )  #inputs, outputs, inputs_test, outputs_test
## XX = list(X_train_mm.itertuples(index=False, name=None))
## print(XX)

### Use for architecture development HPC cluster
##X_train_mm = pd.read_csv('/scratch/mmccormi1/X_train_mm.csv', sep=',', decimal='.', header=None, index_col=False)
##y_train_mm = pd.read_csv('/scratch/mmccormi1/y_train_mm.csv', sep=',', decimal='.', header=None, index_col=False)
##X_test_mm = pd.read_csv('/scratch/mmccormi1/X_test_mm.csv', sep=',', decimal='.', header=None, index_col=False)
##y_test_mm = pd.read_csv('/scratch/mmccormi1/y_test_mm.csv', sep=',', decimal='.', header=None, index_col=False)
##
######### # Use for architecture development PC
##X_train_mm = pd.read_csv('C:/Users/mark_/mark_data/Input/X_train_mm.csv', sep=',', decimal='.', header=None, index_col=False)
##y_train_mm = pd.read_csv('C:/Users/mark_/mark_data/Input/y_train_mm.csv', sep=',', decimal='.', header=None, index_col=False)
##X_test_mm = pd.read_csv('C:/Users/mark_/mark_data/Input/X_test_mm.csv', sep=',', decimal='.', header=None, index_col=False)
##y_test_mm = pd.read_csv('C:/Users/mark_/mark_data/Input/y_test_mm.csv', sep=',', decimal='.', header=None, index_col=False)
########## # h=0
########## # n=0
########    
##from satw_nn_p import AFBR_MLP_archopti   #  If 1-layer model, then comment the multi layer evaluation.
######
####layer = [1,2,3,4,5,6,7,8,9,10,11,12]# 12 + 1 layer run separately = 13 layers on the heatmap
##l=1   # Use for the 1-layer model
##node=  [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]#, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096
######  13 layers x 10 nodes = 130 runs.
##Archopti_test_1_layer = []
####Arch_NN = []
##Archopti_test = []
##Archopti_test_mse = []
##Archopti_mse_eval = []
### h=0#[[0]]
### n=0
##for l in layer:
##    # print(h)
##    for n in node:
##        # print(n)
##        #Yhat_o, Yhat_v_o, test_mse, mse_eval = AFBR_MLP_archopti(X_train_mm.to_numpy(dtype=float), y_train_mm, X_test_mm.to_numpy(dtype=float), y_test_mm, h, n)  #inputs, outputs, inputs_test, outputs_test
##        Yhat_o, Yhat_v_o, test_mse, mse_eval = AFBR_MLP_archopti(X_train_mm, y_train_mm, X_test_mm, y_test_mm, l, n)  #inputs, outputs, inputs_test, outputs_test
##        # Arch_NN.append([h,n])
##        Archopti_test.append(Yhat_v_o)
##        Archopti_test_mse.append(test_mse)
##        Archopti_mse_eval.append(mse_eval)
##
######   USE ONLY WHEN RUNNING ON THE HPC CLUSTERS    ########
##np.savetxt("/scratch/mmccormi1/Archopti_test.csv", np.hstack(Archopti_test), delimiter=",", fmt="%10.4f") 
##np.savetxt("/scratch/mmccormi1/Archopti_test_mse.csv", np.asarray(Archopti_test_mse), delimiter=",", fmt="%10.4f") 
##np.savetxt("/scratch/mmccormi1/Archopti_mse_eval.csv", np.asarray(Archopti_mse_eval), delimiter=",", fmt="%10.4f") 
#### save min-max scaled test response datasets to the cluster
####np.savetxt("/scratch/mmccormi1/y_test_mm.csv", y_test_mm, delimiter=";", fmt="%10.2f") 
####np.savetxt("/scratch/mmccormi1/y_test_mm_4b.csv", y_test_mm, delimiter=";", fmt="%10.2f") 
##
######  For 1-layer model evaluation

##for n in node:
##        # print(n)
##        #Yhat_o, Yhat_v_o, test_mse, mse_eval = AFBR_MLP_archopti(X_train_mm.to_numpy(dtype=float), y_train_mm, X_test_mm.to_numpy(dtype=float), y_test_mm, h, n)  #inputs, outputs, inputs_test, outputs_test
##    Yhat_o, Yhat_v_o, test_mse, mse_eval, Yhat_v_mlp_1_layer = AFBR_MLP_archopti(X_train_mm, y_train_mm, X_test_mm, y_test_mm, l, n)  #inputs, outputs, inputs_test, outputs_test
##        # Arch_NN.append([h,n])
##    Archopti_test_1_layer.append(Yhat_v_mlp_1_layer[:,0])
####    Archopti_test_mse.append(test_mse)
####    Archopti_mse_eval.append(mse_eval)
##
####   USE ONLY WHEN RUNNING ON THE PC    ########
##np.savetxt("C:/Users/mark_/mark_data/Output/Archopti_test_1_layer.csv", np.asarray(Archopti_test_1_layer).T , delimiter=",", fmt="%10.4f")#
## np.savetxt("C:/Users/mark_/userdata/Output/Archopti_test_mse.csv", np.asarray(Archopti_test_mse), delimiter=",", fmt="%1.4f") #, fmt="%10.4f"        
## np.savetxt("C:/Users/mark_/userdata/Output/Archopti_mse_eval.csv", np.asarray(Archopti_mse_eval), delimiter=",", fmt="%1.4f")
## np.savetxt("C:/Users/mark_/userdata/Output/Yhat_v_1l.csv", np.concatenate(Archopti_test, axis=1), delimiter=",", fmt="%1.4f") #, 




#################  COMMENT FROM LINE BELOW WHEN RUNNING ON THE HPC   ################

##    Training and testing data
#  Shuffled data
# X_train_mm = np.genfromtxt(open("C:/Users/mark_/userdata/Output/X_train_mm.csv"), delimiter=',')
# X_test_mm = np.genfromtxt(open("C:/Users/mark_/userdata/Output/X_test_mm.csv"), delimiter=',')
# y_train_mm = np.genfromtxt(open("C:/Users/mark_/userdata/Output/y_train_mm.csv"), delimiter=',')
# y_test_mm = np.genfromtxt(open("C:/Users/mark_/userdata/Output/y_test_mm.csv"), delimiter=',')


# ##   UNSHUFFLED, TIME SERIES DATA
# X_train_mm = np.genfromtxt(open("C:/Users/mark_/userdata/Output/X_train_mm_ns.csv"), delimiter=',')
# X_test_mm = np.genfromtxt(open("C:/Users/mark_/userdata/Output/X_test_mm_ns.csv"), delimiter=',')
# y_train_mm = np.genfromtxt(open("C:/Users/mark_/userdata/Output/y_train_mm_ns.csv"), delimiter=',')
# y_test_mm_ns = np.genfromtxt(open("C:/Users/mark_/userdata/Output/y_test_mm_ns.csv"), delimiter=',')


# ##########  Polynomial    run on the PC         ###################
# # Shuffled data  Yhat_polynomial_m = mean prediction for each row
# ## Use if dataframe needs to be converted to array
# y_train_mm , y_test_mm = y_train_mm.reshape(-1,1), y_test_mm.reshape(-1,1)
# # X_train_mm, , X_test_mm
# # (np.reshape(y_train_mm.to_numpy().astype('float32'), (y_train_mm.shape[0], 1, y_train_mm.shape[1])),
# #     np.reshape(X_train_mm.to_numpy().astype('float32'), (X_train_mm.shape[0], 1, X_train_mm.shape[1])), 
# #     np.reshape(y_test_mm.to_numpy().astype('float32'), (y_test_mm.shape[0], 1, y_test_mm.shape[1])),
# #     np.reshape(X_test_mm.to_numpy().astype('float32'), (X_test_mm.shape[0], 1, X_test_mm.shape[1])))
# # inputs_test_r = np.reshape(inputs_test.to_numpy().astype('float32'), (inputs_test.shape[0], 1, inputs_test.shape[1]))
# z = y_train_mm.reshape(-1,1)   
# y_train_mm_a = np.reshape(y_train_mm.to_numpy().astype('float32'), (y_train_mm.shape[0], 1)) #, y_train_mm.shape[1]

# from satw_data import poly_rep
# Yhat_polynomial_m, rms_poly_mean, R2_poly_mean = poly_rep(y_train_mm, X_train_mm, y_test_mm, X_test_mm)
# # np.savetxt("C:/Users/mark_/userdata/Output/Yhat_poly100_loo.csv",Yhat_polynomial_m , delimiter=",", fmt="%10.4f")#

# # # UNshuffled data  Yhat_polynomial_m = mean prediction for each row
# Yhat_polynomial_m, rms_poly_mean, R2_poly_mean = poly_rep(y_train_mm, X_train_mm, y_test_mm, X_test_mm)
# # Yhat_polynomial_m_ns = Yhat_polynomial_m # The result when using non shuffled data set
# # np.savetxt("C:/Users/mark_/userdata/Output/Yhat_poly100_loo_ns.csv", Yhat_polynomial_m_ns , delimiter=",", fmt="%1.4f")  #np.asarray(Archopti_test_mse)

# # y_test_mm = y_test_mm_ns
# # plt.scatter(outputs_test,Yhat_v)
# rms_poly = sqrt(mean_squared_error(y_test_mm, Yhat_polynomial_m))
# print("Polynomial model rms error is: " + str(rms_poly))

# reg_poly = LinearRegression().fit(y_test_mm.reshape(-1,1),Yhat_polynomial_m.reshape(-1,1)) #, fit_intercept=True
# print("Polynomial model validation test R2: %.2f" % reg_poly.score(y_test_mm.reshape(-1,1),Yhat_polynomial_m.reshape(-1,1)))  

# ####   MLP    run on the PC    or HPC        ######################
# ###shuffled data
# import satw_mlp_v5
# # from satw_mlp_v5 import AFBR_MLP_model

# # Shuffled data
# Yhat, Yhat_v = satw_mlp_v5.AFBR_MLP_model(X_train_mm, y_train_mm, X_test_mm, y_test_mm) #inputs, outputs, inputs_test, outputs_test
# # # # np.savetxt("C:/Users/mark_/userdata/Output/Yhat_v_mlp.csv", Yhat_v , delimiter=",", fmt="%10.4f")#

# # plt.scatter(outputs_test,Yhat_v)
# rms_mlp = sqrt(mean_squared_error(y_test_mm,Yhat_v))
# print("MLP model rms error is: " + str(rms_mlp))

# reg_mlp = LinearRegression().fit(np.asarray(y_test_mm).reshape(-1,1),Yhat_v.reshape(-1,1)) #, fit_intercept=True
# print("MLP model validation test R2: %.2f" % reg_mlp.score(np.asarray(y_test_mm).reshape(-1,1),Yhat_v.reshape(-1,1)))    

# # ## UN Shuffled data
# Yhat, Yhat_v = satw_mlp_v5.AFBR_MLP_model(X_train_mm, y_train_mm, X_test_mm, y_test_mm_ns) #inputs, outputs, inputs_test, outputs_test
# np.savetxt("C:/Users/mark_/userdata/Output/Yhat_v_mlp_ns.csv", Yhat_v , delimiter=",", fmt="%10.4f")#

# y_test_mm = y_test_mm_ns
# # plt.scatter(outputs_test,Yhat_v)
# rms_mlp = sqrt(mean_squared_error(y_test_mm,Yhat_v))
# print("MLP model rms error is: " + str(rms_mlp))

# reg_mlp = LinearRegression().fit(y_test_mm.reshape(-1,1),Yhat_v.reshape(-1,1)) #, fit_intercept=True
# print("MLP model validation test R2: %.2f" % reg_mlp.score(y_test_mm.reshape(-1,1),Yhat_v.reshape(-1,1)))    

# ###########   LSTM   run on the PC    ###################
##   UNSHUFFLED, TIME SERIES DATA
##  USE the same data as for polynomial and MLP

# from satw_mlp_v5 import AFBR_lstm

# ###  Shuffled data
# inputs_train, outputs_train, inputs_test, outputs_test = X_train_mm, y_train_mm, X_test_mm, y_test_mm
# Yhat, Yhat_v= AFBR_lstm(inputs_train, outputs_train, inputs_test, outputs_test)

# # np.savetxt("C:/Users/mark_/userdata/Output/Yhat_v_lstm_mm.csv", Yhat_v, delimiter=",", fmt="%1.4f")


# ###  Not shuffled data
# # inputs_train, outputs_train, inputs_test, outputs_test = X_train_mm, y_train_mm, X_test_mm, y_test_mm_ns
# # Yhat, Yhat_v= AFBR_lstm(inputs_train, outputs_train, inputs_test, outputs_test)

# # np.savetxt("C:/Users/mark_/userdata/Output/Yhat_v_lstm_mm_ns.csv", Yhat_v, delimiter=",", fmt="%1.4f")


# # plt.scatter(outputs_test,Yhat_v)
# rms_lstm = sqrt(mean_squared_error(outputs_test,Yhat_v))
# print("LSTM model rms error is: " + str(rms_lstm))

# reg = LinearRegression().fit(outputs_test.reshape(-1,1),Yhat_v.reshape(-1,1)) #, fit_intercept=True
# print("LSTM model validation test R2: %.2f" % reg.score(outputs_test.reshape(-1,1),Yhat_v.reshape(-1,1)))    

# reg = LinearRegression().fit(outputs_test, Yhat_v) #, fit_intercept=True
# print("LSTM model validation test R2: %.2f" % reg.score(outputs_test, Yhat_v))

##############################################################################
# ###############        MAKE POLYNOMIAL REGRESSION PLOTS FOR Final paper    ####################

#### Open all saved results and make a single data frame

######   Shuffled data
##y_test_mm = np.genfromtxt(open("C:/Users/mark_/mark_data/Output/y_test_mm.csv"), delimiter=',')
######y_test_lstm = np.genfromtxt(open("C:/Users/mark_/mark_data/Output/V24/y_test_lstm.csv"), delimiter=',')
##Yhat_v_poly = np.genfromtxt(open("C:/Users/mark_/mark_data/Output/Yhat_v_poly.csv"), delimiter=',')
##Yhat_v_mlp = np.genfromtxt(open("C:/Users/mark_/mark_data/Output/V24/Yhat_v_mlp.csv"), delimiter=',')
####
####print('y_test_mm', y_test_mm)
##print('Yhat_v_poly shape', Yhat_v_poly.shape)
####print('Yhat_v_poly', Yhat_v_poly)
####Yhat_v_mlp = Yhat_v_mlp.T #np.rot90(Yhat_v_mlp, k=1, axes=(0,1))
##print('Yhat_v_mlp', Yhat_v_mlp)
##print('Yhat_v_mlp shape', Yhat_v_mlp.shape)
####
##Yhat_v_lstm = np.genfromtxt(open("C:/Users/mark_/mark_data/Output/V24/Yhat_v_lstm.csv"), delimiter=',')
##print('Yhat_v_lstm', Yhat_v_lstm)
##print('Yhat_v_lstm shape', Yhat_v_lstm.shape)
####
########   Non-shuffled data
##y_test_mm = np.genfromtxt(open("C:/Users/mark_/mark_data/Output/V24/y_test_mm_ns.csv"), delimiter=',')
####y_test_lstm = np.genfromtxt(open("C:/Users/mark_/mark_data/Output/V24/y_test_lstm_ns.csv"), delimiter=',')
##Yhat_v_poly = np.genfromtxt(open("C:/Users/mark_/mark_data/Output/V24/Yhat_v_poly_ns.csv"), delimiter=',')
##Yhat_v_mlp = np.genfromtxt(open("C:/Users/mark_/mark_data/Output/V24/Yhat_v_mlp_ns.csv"), delimiter=',')
##Yhat_v_lstm = np.genfromtxt(open("C:/Users/mark_/mark_data/Output/V24/Yhat_v_lstm_ns.csv"), delimiter=',')
##
#####print('Yhat_v_poly matrix mean', Yhat_v_poly.mean(axis=1))
####print('Yhat_v_mlp matrix mean', Yhat_v_mlp)
##
###data = np.concatenate((Yhat_v_poly.mean(axis=1),Yhat_v_mlp), axis=0) #y_test_mm, 
####cols_306 = ['y_test_mm','Yhat_v_poly','Yhat_v_mlp'] #'y_test_mm',
####Model_306 = pd.DataFrame(data=(y_test_mm.T, Yhat_v_poly.mean(axis=1).T, Yhat_v_mlp.T)).T
####Model_306.columns = cols_306
#####Model_306.describe()
####Model_306.to_csv("C:/Users/mark_/mark_data/Output/V24/Model_306.csv", sep=',', float_format='%.3f', header=True, decimal=',')
####
######cols_340 = ['Yhat_v_lstm','Yhat_v_lstm'] 
######Model_340 = pd.DataFrame(data = [Yhat_v_lstm.T, Yhat_v_lstm.T], columns=cols_340)
######Model_340.columns = cols_340
######Model_340.describe()
######Model_340.to_csv("C:/Users/mark_/mark_data/Output/V24/Model_340.csv", sep=',', float_format='%.3f', header=True, decimal=',')
####
####print('lenght',len(Model_306.iloc[0]))
####print(len(Model_306.iloc[1]))

####Plot a slope of 1 to show a perfect correlation between predictions and true values
##yl = np.array([0,1,0.1])
##xl = np.array([0,1, 0.1])
##
##nrep=100
##
###plt.figure(figsize=(14.1,10))  # Shuffled data. Figure in Sustainability
##fig, axs = plt.subplots(figsize=(16.2,10))
##plt.plot(xl,yl,  c="k")
##for p in range(nrep):
##    plt.scatter(y_test_mm, Yhat_v_poly[:,p],  c="b", s=3, label="Polynomial model")
##for m in range(nrep):
##    plt.scatter(y_test_mm, Yhat_v_mlp[:,m],  c="r", s=3, label = "MLP model")
####for l in range(nrep):
####    plt.scatter(y_test_lstm, Yhat_v_lstm,  c="g", s=3, label = "LSTM model")
####plt.plot(y_test_mm, color= "r", linewidth=1, label = "True values")
##plt.xlabel('True values') #Measurement sequence number
##plt.ylabel('Predictions')
##axs.tick_params(axis="y", labelsize=16, direction="in")
##axs.set_ylabel("Predictions", fontsize=24)
##axs.tick_params(axis="x", labelsize=16, direction="in")
##axs.set_xlabel("True values", fontsize=24)
###plt.legend()
###plt.title('Comparison of Polynomial, MLP, and LSTM models, Validation test, pre-shuffled data', fontsize='large') #MLP predicted and observed flow rate during testing and LSTM
####plt.savefig('C:/Users/mark_/Documents/02_Professional/03_Unil/SATW/MDPI_paper/Journal communication/Sustainability/Round two special issue/Figures_originals/Fig_PvT_shuffled.svg', format="svg")
####plt.savefig('C:/Users/mark_/Documents/02_Professional/03_Unil/SATW/MDPI_paper/Journal communication/Sustainability/Round two special issue/Figures_originals/Fig_PvT_shuffled.pdf', format="pdf", bbox_inches="tight", dpi=300)
##plt.savefig('C:/Users/mark_/Documents/02_Professional/03_Unil/SATW/MDPI_paper/Journal communication/Sustainability/Round two special issue/Figures_originals/Fig_PvT_not_shuffled.svg', format="svg")
##plt.savefig('C:/Users/mark_/Documents/02_Professional/03_Unil/SATW/MDPI_paper/Journal communication/Sustainability/Round two special issue/Figures_originals/Fig_PvT_not_shuffled.pdf', format="pdf", bbox_inches="tight", dpi=300)
####plt.savefig('C:/Users/mark_/Documents/02_Professional/03_Unil/SATW/MDPI_paper/complementary_information/figures/Fig_PvT_shuffled.svg', format="svg")
##plt.show()


##print('y_test_mm', y_test_mm)
##print('Yhat_v_mlp_ns', Yhat_v_mlp_ns)
###print('Yhat_v_mlp_ns.mean(axis=1)', np.mean(Yhat_v_mlp_ns[:,:], axis=1))
##A = Yhat_v_mlp_ns.flatten().reshape(len(y_test_mm),-1)
##print(np.mean(A, axis=1))
##
## axs.plot(y, linestyle=" ", marker='D', c="r",  markersize=5, label=my_label["Reference experiment"])  # Reference experiencekind='scatter',ax=axs,
##    axs.tick_params(axis="y", labelsize=16, direction="in")
##    axs.set_ylabel("Calorific value reduction [J.liter$^{-1}$]", fontsize=24)
##    
##    axs.plot(CV_red_n, linestyle=" ", marker='.', c="b", markersize=5, label=my_label["Derived values"])
##    axs.tick_params(axis="x", labelsize=16, labelrotation=-90, direction="in")
##    axs.set_xlabel("Date", fontsize=24)#
##    plt.subplots_adjust(bottom=0.175)

##### Non shuffled data. Figure in Sustainability
#plt.figure(figsize=(16.2,10))
##fig, axs = plt.subplots(figsize=(16.2,10))
###plt.scatter(Model_306.loc['y_test_mm'], Model_306.loc['Yhat_v_poly'])
###plt.scatter(Model_306.iloc[0], Model_306.iloc[1])
###Model_306[cols_306].plot(style=".")#('y_test_mm','Yhat_v_poly'), kind='scatter')
##plt.plot(xl,yl,  c="k")
##for p in range(nrep):
##    plt.scatter(y_test_mm, Yhat_v_poly[:,1],  c="b", s=3)#, label="Polynomial model")
##for m in range(nrep):
##    plt.scatter(y_test_mm, Yhat_v_mlp[:,1],  c="r", s=3)#, label = "MLP model")
####for m in range(nrep):
####    plt.scatter(y_test_lstm, Yhat_v_lstm,  c="g", s=3, label="LSTM model")
####plt.scatter(y_test_mm, np.mean(Yhat_v_mlp_ns.flatten().reshape(len(y_test_mm),-1),axis=1),  c="r", s=5, label = "MLP model")
###plt.scatter(y_test_lstm_ns, Yhat_v_lstm_ns,  c="g", s=5, label = "LSTM model")
####plt.plot(y_test_mm, color= "r", linewidth=1, label = "True values")
####plt.xlabel('True values') #Measurement sequence number
####plt.ylabel('Predictions')
##axs.tick_params(axis="y", labelsize=16, direction="in")
##axs.set_ylabel("Predictions", fontsize=24)
##axs.tick_params(axis="x", labelsize=16, direction="in")
##axs.set_xlabel("True values", fontsize=24)
##   
####plt.legend(["","Polynomial model", "MLP model"])
####plt.title('Comparison of Polynomial,MLP, and LSTM models, Validation test, Not-shuffled data', fontsize='large') #MLP predicted and observed flow rate during testing
###plt.savefig('C:/Users/mark_/Documents/02_Professional/03_Unil/SATW/MDPI_paper/Journal communication/Sustainability/Round two special issue/Figures_originals/Fig_PvT_not_shuffled.svg', format="svg")
####plt.savefig('C:/Users/mark_/Documents/02_Professional/03_Unil/SATW/MDPI_paper/complementary_information/figures/Fig_PvT_not_shuffled.svg', format="svg")
##plt.show()    

# Yhat_v_polynomial_m = np.genfromtxt(open("C:/Users/mark_/userdata/Output/Yhat_poly100_loo.csv"), delimiter=',')
# Yhat_v_mlp = np.genfromtxt(open("C:/Users/mark_/userdata/Output/Yhat_v_mlp.csv"), delimiter=',')
# Yhat_v_lstm_mm = np.genfromtxt(open("C:/Users/mark_/userdata/Output/Yhat_v_lstm_mm.csv"), delimiter=',')

# ###### Shuffled data, Compare polynomial, ANN and MLP models   ########
# #F7_poly_ann
# # x = np.array([x for x in range(len(y_test_mm))])
# # Err = y_test_mm-Yhat_v
# # Err = np.subtract(y_test_mm,Yhat_v)
# # Errb = np.add(y_test_mm, Err)
# # plt.legend()#['True outputs (Test data)', 'Predictions (Test data)', 'Predictions (Test-2 data)'])
# ## Results when using shuffled data
# plt.figure(figsize=(8,8)) 
# plt.plot(xl,yl,  c="k")
# plt.scatter(y_test_mm, Yhat_v_polynomial_m,  c="b", s=2, label="Polynomial model")
# plt.scatter(y_test_mm, Yhat_v_mlp,  c="r", s=1, label = "MLP model")
# plt.scatter(y_test_mm, Yhat_v_lstm_mm,  c="g", s=2, label = "LSTM model")
# # plt.plot(y_test_mm, color= "r", linewidth=1, label = "True values")

# # plt.plot(yhat_r, linestyle="-", c="b", linewidth=1, label = "linear regression - Predictions")
# # plt.plot(Err, label = "Raw error")
# # plt.plot(Errb[:,0], color ="#444444", label = "Errb")
# # plt.plot(Yhat_v, label = "Preditions using MLP")
# # plt.plot(y_test_mm + Err, label = "True values + prediction error")

# plt.xlabel('True values') #Measurement sequence number
# plt.ylabel('Predictions')
# plt.legend()
# plt.title('Comparison of Polynomial, MLP and LSTM models, Validation test, pre-shuffled data', fontsize='large') #MLP predicted and observed flow rate during testing
# plt.savefig('C:/Users/mark_/Documents/02_Professional/03_Unil/SATW/MDPI_paper/MDPI_article/MDPI_article_v21/figures/Fig_PvT_shuffled.svg', format="svg")
# plt.show();    

# ###   UN Shuffled data    ns = non-shuffled
# y_test_mm_ns = np.genfromtxt(open("C:/Users/mark_/userdata/Output/y_test_mm_ns.csv"), delimiter=',')
# Yhat_v_polynomial_m_ns = np.genfromtxt(open("C:/Users/mark_/userdata/Output/Yhat_poly100_loo_ns.csv"), delimiter=',')
# Yhat_v_mlp_ns = np.genfromtxt(open("C:/Users/mark_/userdata/Output/Yhat_v_mlp_ns.csv"), delimiter=',')
# Yhat_v_lstm_mm_ns = np.genfromtxt(open("C:/Users/mark_/userdata/Output/Yhat_v_lstm_mm_ns.csv"), delimiter=',')

# ###### UNShuffled data, Compare polynomial, ANN and MLP models   ########
# # #F7_poly_ann
# # # x = np.array([x for x in range(len(y_test_mm))])
# # # Err = y_test_mm-Yhat_v
# # # Err = np.subtract(y_test_mm,Yhat_v)
# # # Errb = np.add(y_test_mm, Err)
# # # plt.legend()#['True outputs (Test data)', 'Predictions (Test data)', 'Predictions (Test-2 data)'])
# # ## Results when using shuffled data
# plt.figure(figsize=(8,8)) 
# plt.plot(xl,yl,  c="k")
# plt.scatter(y_test_mm_ns, Yhat_v_polynomial_m_ns,  c="b", s=2, label="Polynomial model")
# plt.scatter(y_test_mm_ns, Yhat_v_mlp_ns,  c="r", s=1, label = "MLP model")
# plt.scatter(y_test_mm_ns, Yhat_v_lstm_mm_ns,  c="g", s=2, label = "LSTM model")
# # plt.plot(y_test_mm, color= "r", linewidth=1, label = "True values")

# # plt.plot(yhat_r, linestyle="-", c="b", linewidth=1, label = "linear regression - Predictions")
# # plt.plot(Err, label = "Raw error")
# # plt.plot(Errb[:,0], color ="#444444", label = "Errb")
# # plt.plot(Yhat_v, label = "Preditions using MLP")
# # plt.plot(y_test_mm + Err, label = "True values + prediction error")

# plt.xlabel('True values') #Measurement sequence number
# plt.ylabel('Predictions')
# plt.legend()
# plt.title('Comparison of Polynomial, MLP and LSTM models, Validation test, unshuffled data', fontsize='large') #MLP predicted and observed flow rate during testing
# plt.savefig('C:/Users/mark_/Documents/02_Professional/03_Unil/SATW/MDPI_paper/MDPI_article/MDPI_article_v21/figures/Fig_PvT_unshuffled.svg', format="svg")
# plt.show();    


# # y_test_mm = pd.read_csv('C:/Users/mark_/userdata/Output/y_test_mm.csv', sep=',', decimal='.', header=None, index_col=False)
# # # y_test_mm_4c = pd.read_csv('C:/Users/mark_/userdata/Output/y_test_mm_4c.csv', sep=',', decimal='.', header=None, index_col=False)

# # from satw_data import satw_figures#, poly_rep
# # satw_figures(y_test_mm)#, y_test_mm_4c)# Yhat_polynomial)#CV_ref_all, IFR_ref_all, 


# ###############        MAKE STATISTICS     ####################
# from satw_data import satw_stats
# satw_stats(y_train_mm, y_test_mm)


# # ##### Run the 1-layer model
# # # import satw_1layer
# # # X_train_mm, y_train_mm, X_test_mm, y_test_mm = inputs, outputs, inputs_test, outputs_test 
# # # satw_1layer.AFBR_1l_model(X_train_mm, y_train_mm, X_test_mm, y_test_mm) #
# # # satw_1layer.AFBR_1l_sim(SX_test_mm)
# # # AFBR_1l_model(X_train_mm, y_train_mm, X_test_mm, y_test_mm)


# # ######## Testing during model building, Single run   #################### 

# # # AFBR = keras.models.load_model('C:/Users/mark_/Anaconda3/envs/Tensorflow_v2/Mark/AFBR_mlp_test.h5') #  AFBR_1_test
# # # #AFBR = keras.models.load_model('/scratch/mmccormi1/AFBR_mlp_test.h5') #  AFBR_1_test
# # # Yhat = AFBR.predict([X_train_mm])# training
# # # Yhat_v = AFBR.predict([X_test_mm])# Validation test predictions
# # # print('Yhat', Yhat.shape, 'Yhat_t', Yhat_t.shape,'X_test_mm', X_test_mm.shape)

# # #######     MDPI figures 4a, 4b, 4c and 4d    OLD ARCHIVES DO NOT USE  #########
# #     # fig 4a = 1-layer perceptron with shuffled data (for code see next section and file satw_mlp_v5)
# #     # fig 4b = MLP with unshuffled data (set shuffle to False in NN_preprocess function, satw_data file)
# #     # fig 4c = MLP with pre-shuffled data (set shuffle to True in NN_preprocess function, satw_data file)
# #     # fig 4d = MLP with pre-shuffled data and gradual reduction in the number of hidden units from layer 6 to the output layer
   

# # # # # Manually create arrays for use in figures

# # # # y_test_mm_4a = y_test_mm
# # # # Yhat_v_4a = np.mean(Yhat_v_rep, axis = 0)

# # # # # y_test_mm_4b = y_test_mm
# # # # # Yhat_v_4b = np.mean(Yhat_v_rep, axis = 0)

# # # y_test_mm_4c = y_test_mm
# # # Yhat_v_4c =  np.mean(Yhat_v_rep, axis = 0)

# # # # # y_test_mm_4d = y_test_mm
# # # # # Yhat_v_4d = np.mean(Yhat_v_rep, axis = 0)


# F4_ABCD, ((axa, axb), (axc, axd)) = plt.subplots(2, 2, sharex=True, sharey=True)

# # # plt.savefig('C:/Users/mark_/Documents/02_Professional/03_Unil/SATW/MDPI_paper/MDPI_article/MDPI_article_v11/figures/F4_ABCD.svg', format="svg") #F4_ABCD
# # plt.savefig('/scratch/mmccormi1/F4_ABCD.svg', format="svg")

# # #### MDPI figure 4a  (1-layer NN)    #######

# # reg = LinearRegression().fit(Yhat_v_4a, y_test_mm_4a) #, fit_intercept=True
# # # # print("Validation test R2: %.2f" % reg.score(Yhat_v, y_test_mm))    
# # # # # plt.figure(figsize=(7,6)) 
# # # # plt.subplot(221)
# # axa.scatter(y_test_mm_4a, Yhat_v_4a,  c="k", s=1)
# # # # plt.plot(y_pred_v)
# # axa.plot(xl,yl,  c="k", linestyle='solid') #dashed
# # axa.set_title('4a 1-layer perceptron', fontsize=10)
# # # axa.annotate("R2: %.2f" % reg.score(Yhat_v_4a, y_test_mm_4a), (10,10))
# # # # plt.legend()#['True outputs (Test data)', 'Predictions (Test data)', 'Predictions (Test-2 data)'])
# # plt.xlabel('True values') #Measurement sequence number
# # plt.ylabel('Predictions')
# # plt.title('l-layer perceptron model - predicted and true values, validation test, daily CV reduction (MinMax scaled)', fontsize='large') #MLP predicted and observed flow rate during testing
# # # # plt.savefig('C:/Users/mark_/Documents/02_Professional/03_Unil/SATW/MDPI_paper/MDPI_article/MDPI_article_v09/figures/F4_A_1-layer_test_shuffled.svg', format="svg")
# # plt.show() 


# # print("4a, validation test R2: %.2f" % reg.score(Yhat_v_4a, y_test_mm_4a))


# # ##### MDPI figure 4b - MLP with unshuffled data
# # # To make this figure do preprocessing separately.
# # # Yhat_v is returned from the satw_mlp_v5 file
# # # plt.figure(figsize=(7,6))  
# # # plt.subplot(222) 
# # # axb.scatter(y_test_mm_4b, Yhat_v_4b,  c="k", s=1)
# # # axb.plot(xl,yl,  c="k")
# # # axb.set_title('4b 6-layer perceptron, not shuffled', fontsize=10)
# # # # # plt.legend()#['True outputs (Test data)', 'Predictions (Test data)', 'Predictions (Test-2 data)'])
# # # # plt.xlabel('True values') #Measurement sequence number
# # # # plt.ylabel('Predictions')
# # # # plt.title('MLP model - unshuffled data, predicted and true values, validation test  Daily CV reduction (MinMax scaled)', fontsize='large') #MLP predicted and observed flow rate during testing
# # # # # plt.savefig('C:/Users/mark_/Documents/02_Professional/03_Unil/SATW/MDPI_paper/MDPI_article/MDPI_article_v09/figures/F4_B_6-layer-MLP_test_unshuffled.svg', format="svg")
# # # # plt.show()

# # # reg = LinearRegression().fit(Yhat_v_4b, y_test_mm_4b) #, fit_intercept=True
# # # print("4b, Validation test R2: %.2f" % reg.score(Yhat_v_4b, y_test_mm_4b))


# ##### MDPI figure 4c - MLP with preshuffled data
# # To make this figure do preprocessing separately.
# # Yhat_v is returned from the satw_mlp_v5 file
# #plt.figure(figsize=(7,6))   
# axc.scatter(y_test_mm_4c, Yhat_v_4c, linestyle='-', c="k", s=1)
# axc.plot(xl,yl,  c="k")
# axc.set_title('4c 6-layer perceptron, shuffled', fontsize=10)
# # plt.legend()#['True outputs (Test data)', 'Predictions (Test data)', 'Predictions (Test-2 data)'])
# plt.xlabel('True values') #Measurement sequence number
# plt.ylabel('Predictions')
# # plt.title('MLP model - preshuffled data, predicted and true values, validation test  Daily CV reduction (MinMax scaled)', fontsize='large') #MLP predicted and observed flow rate during testing
# # # plt.savefig('C:/Users/mark_/Documents/02_Professional/03_Unil/SATW/MDPI_paper/MDPI_article/MDPI_article_v09/figures/F4_C_6-layer-MLP_test_preshuffled.svg', format="svg")
# # plt.show()

# reg = LinearRegression().fit(Yhat_v_4c, y_test_mm_4c) #, fit_intercept=True
# print("4c, Validation test R2: %.2f" % reg.score(Yhat_v_4c, y_test_mm_4c))

# # ##### MDPI figure 4d - MLP with preshuffled data and gradual reduction in the number of hidden layers   ########
# # # To make this figure do preprocessing separately.
# # # Yhat_v is returned from the satw_mlp_v5 file
# # # plt.figure(figsize=(7,6))   
# # axd.scatter(y_test_mm_4d, Yhat_v_4d,  c="k", s=1)
# # axd.plot(xl,yl,  c="k")
# # axd.set_title('4d 11-layer MLP, grad red in num units per layer', fontsize=10)
# # # plt.legend()#['True outputs (Test data)', 'Predictions (Test data)', 'Predictions (Test-2 data)'])
# # # plt.xlabel('True values') #Measurement sequence number
# # # plt.ylabel('Predictions')
# # # plt.title('MLP model - preshuffled data, gradual reduction in nimber of hidden layers, predicted and true values, validation test  Daily CV reduction (MinMax scaled)', fontsize='large') #MLP predicted and observed flow rate during testing
# # # plt.savefig('C:/Users/mark_/Documents/02_Professional/03_Unil/SATW/MDPI_paper/MDPI_article/MDPI_article_v09/figures/F4_D_6-layer-MLP_test_preshuffled_gradual.svg', format="svg")
# # # plt.show()

# # reg = LinearRegression().fit(Yhat_v_4d, y_test_mm_4d) #, fit_intercept=True
# # print("4d, Validation test R2: %.2f" % reg.score(Yhat_v_4d, y_test_mm_4d))

# # # plt.savefig('C:/Users/mark_/Documents/02_Professional/03_Unil/SATW/MDPI_paper/MDPI_article/MDPI_article_v11/figures/F4_ABCD.svg', format="svg") #F4_ABCD
##############  COMMENT ABOVE WHEN RUNNING HPC ##################


####################  LINEAR REGRESSION   #########################

# reg = LinearRegression().fit(Yhat, y_train_mm) #, fit_intercept=True
# y_pred = reg.predict(y_train_mm)#, color='blue', linewidth=3
# r2 = "R2: %.2f" % reg.score(Yhat, y_train_mm)
# print("train R2: %.2f" % reg.score(Yhat, y_train_mm))

# reg = LinearRegression().fit(Yhat_v, y_test_mm) #, fit_intercept=True
# y_pred = reg.predict(y_test_mm)#, color='blue', linewidth=3
# r2 = "R2: %.2f" % reg.score(Yhat_v, y_test_mm)
# print("test R2: %.2f" % reg.score(Yhat_v, y_test_mm))



# ######### For information Not in MDPI article  ###################

# plt.plot(yhat_regression[1])
# yl = np.array([0,1,0.1])
# xl = np.array([0,1, 0.1])
# #F5_train_TvsP
# plt.figure(figsize=(7,6)) 
# plt.scatter(yhat_regression[1],y_test_mm, c="k")
# plt.scatter(P,T,  c="m")
# # plt.plot(y_pred)
# plt.plot(xl,yl, c="k")
# # plt.annotate(r2, xy=(10, 10))
# # plt.legend()#['True outputs (Test data)', 'Predictions (Test data)', 'Predictions (Test-2 data)'])
# plt.xlabel('True values') #Measurement sequence number
# plt.ylabel('Predictions')
# # plt.title('MLP model - regression of validation test predicted and true values, Daily CH4 production (MinMax scaled)', fontsize='large') #MLP predicted and observed flow rate during testing
# # plt.savefig('C:/Users/mark_/Documents/02_Professional/03_Unil/SATW/MDPI_paper/MDPI_article/MDPI_article_v03/figures/F5_mlp_train_TvsP-grad.svg', format="svg")
# plt.show()   

# #####  Test  ###### 
# #     Yhat, Yhat_v = satw_mlp_v5.AFBR_MLP_model(X_train_mm, y_train_mm, X_test_mm, y_test_mm)
# #     Yhat_rep.append(Yhat)
# #     Yhat_v_rep.append(Yhat_v)
# #     Yhat = []  
# #     Yhat_v = []
# #     print("number of reps:", i)
# # mdlr = LinearRegression().fit(X,y)
# # Wr = mdlr.coef_
# # print('Model regression coefficients:',mdlr.coef_)
# # Ir = mdlr.intercept_
# # print('y-intercept:', mdlr.intercept_)
# # Sr = mdlr.score(X, y)
# # print('Coefficient of determination (R^2):', Sr)

# # Yhat_rep = []
# # Yhat_v_rep = []
# # nrep = 6

# # for i in range(nrep):
# #     Yhat, Yhat_v = satw_mlp_v5.AFBR_MLP_model(X_train_mm, y_train_mm, X_test_mm, y_test_mm)
# #     Yhat_rep.append(Yhat)
# #     Yhat_v_rep.append(Yhat_v)
# #     Yhat = []  
# #     Yhat_v = []
# #     print("number of reps:", i)

# # Yhat_train_all= np.concatenate([np.array(j) for j in Yhat_rep], axis = 1) #Yhat_rep #
# # Yhat_test_all= np.concatenate([np.array(j) for j in Yhat_v_rep], axis = 1) #Yhat_v_rep #

# # np.savetxt("C:/Users/mark_/Documents/02_Professional/03_Unil/SATW/MDPI_paper/MDPI_article/MDPI_article_v03/figures/Yhat_train_all", Yhat_train_all, delimiter=',')
# # np.savetxt("C:/Users/mark_/Documents/02_Professional/03_Unil/SATW/MDPI_paper/MDPI_article/MDPI_article_v03/figures/Yhat_test_all", Yhat_test_all, delimiter=',')

# # np.savetxt("C:/Users/mark_/Documents/02_Professional/03_Unil/SATW/MDPI_paper/MDPI_article/MDPI_article_v03/figures/y_train_mm", y_train_mm, delimiter=',')
# # np.savetxt("C:/Users/mark_/Documents/02_Professional/03_Unil/SATW/MDPI_paper/MDPI_article/MDPI_article_v03/figures/y_test_mm", y_test_mm, delimiter=',')

#############################################################################################
##############  HYPERPARAMETER EVALUATION  ##################################################
####### Build the 1-layer or multi-layer MLP model and run it i times  ##################
#C:\Users\mark_\Anaconda3\envs\Tensorflow_v2\Lib\site-packages
# import satw_mlp_v5
# # import sys
# inputs, outputs, inputs_test, outputs_test = X_train_mm, y_train_mm, X_test_mm, y_test_mm
# # # momentum = sys.argv[1]

 # #### Use code below to evaluate hyperparameters
 # # lr = 0.0001 #[0.1, 0.001, 0.00000001]#, 0.0000001, 0.000001]
 # mom = [0.9, 0.5, 0.1] #momentum default = 0
 # # rho = [0.9, 0.99, 0.3] # Discounting factor for the history/coming gradient. Defaults to 0.9.
 # # # sdv = [0.3, 0.2, 0.1]
##bch = [16, 12, 8, 4, 3, 2, 1] #V1 best = 4
## # node = [8, 16, 32, 64, 128, 256, 512, 1024, 2048] # best = 16
## # # # # # # ep = [1000, 500, 250] #A small constant for numerical stability. Defaults to 1e-7 
## # # # # # # epsi = [0.00000001, 0.0001, 0.1]
## # # # # # # b1 = [0.999, 0.9, 0.1]
## # do = [0.01, 0.03, 0.05] # 
## # # # # # # centered = [True, False]
##Yhat_rep = []
##Yhat_v_rep = []
##test_mse_rep = []
##test_rms_rep = []
##
##rep=10
##
##for i in range(len(bch)):
##     # pass
##    Yhat, Yhat_v_mlp, rmse_mlp, r_value = AFBR_MLP_model(X_train_mm, y_train_mm, X_test_mm, y_test_mm, bch[i])# mom[i]
##    Yhat_rep.append(Yhat)
##    Yhat_v_rep.append(Yhat_v_mlp)
####    test_mse_rep.append(test_mse)
##    test_rms_rep.append(rmse_mlp)
##
##plt.figure()
##plt.plot(np.asarray(Yhat_v_rep).flatten().reshape(-1,len(bch)))
##plt.show()
##
##plt.figure()
##plt.plot(np.asarray(test_rms_rep).flatten())#.reshape(-1,len(bch)))
##plt.show()
##
### # tester = np.concatenate(Yhat_v_rep, axis=0).reshape(len(node), -1).T
### # ##  ATTENTION : Use the right file name!
##np.savetxt('C:/Users/mark_/mark_data/Output/Yhat_v_hypereval.csv', np.asarray(Yhat_v_rep).reshape(-1,len(bch)), delimiter=",", fmt="%10.4f")# np.mean(np.concatenate(Yhat_v_rep, axis=0).reshape(len(node), -1).T, delimiter=",", fmt="%10.4f") # Yhat_v_rep
### # np.savetxt('C:/Users/mark_/userdata/Output/test_mse_rep.csv', test_mse_rep, delimiter=",", fmt="%10.4f") # 
# # np.savetxt('C:/Users/mark_/userdata/Output/test_rms_rep.csv', test_rms_rep, delimiter=",", fmt="%10.4f") # 

# np.savetxt("/scratch/mmccormi1/Yhat_rep.csv", np.asarray(Yhat_rep).reshape(-1,rep), delimiter=',')
# np.savetxt("/scratch/mmccormi1/Yhat_v_rep.csv", np.asarray(Yhat_v_rep).reshape(-1,rep), delimiter=',')
# np.savetxt("/scratch/mmccormi1/test_mse_rep.csv", np.asarray(test_mse_rep).reshape(-1,rep), delimiter=',')
# np.savetxt("/scratch/mmccormi1/test_rms_rep.csv", np.asarray(test_rms_rep).reshape(-1,rep), delimiter=',')


# # # # #     # satw_mlp_v5.AFBR_1_layer_model(X_train_mm, y_train_mm, X_test_mm, y_test_mm, ep[i])



########## select MLP or 1-layer model   #########################
# import satw_mlp_v5
# Yhat, Yhat_v = satw_mlp_v5.AFBR_MLP_model(X_train_mm, y_train_mm, X_test_mm, y_test_mm) #inputs, outputs, inputs_test, outputs_test

# # Yhat, Yhat_v = satw_mlp_v5.AFBR_1_layer_model(X_train_mm, y_train_mm, X_test_mm, y_test_mm) #, node[i inputs, outputs, inputs_test, outputs_test 

# np.savetxt("/scratch/mmccormi1/Yhat.csv", Yhat, delimiter=',')
# np.savetxt("/scratch/mmccormi1/Yhat_v.csv", Yhat_v, delimiter=',')


#################  Build and run the LSTM model    ################
# MUST use Tensorflow_v1 for the LSTM model (only). Use Tensorflow_v2 for all other models
# MUST set shuffle = False in satw_data.NN_preprocess
# In code pane type print(tf.__version__)to get the version
# When working with LSTM use inputs_train, outputs_train, inputs_test, outputs_test
# import satw_lstm_tf

# import satw_mlp_v5
# from satw_mlp_v5 import AFBR_lstm

# inputs_train = pd.read_csv('C:/Users/mark_/userdata/Output/X_train_mm_ns.csv', sep=';', decimal='.', header=None, index_col=False).to_numpy()
# inputs_test = pd.read_csv('C:/Users/mark_/userdata/Output/X_test_mm_ns.csv', sep=';', decimal='.', header=None, index_col=False).to_numpy()
# outputs_train = pd.read_csv('C:/Users/mark_/userdata/Output/y_train_mm_ns.csv', sep=';', decimal='.', header=None, index_col=False).to_numpy()   #, dtype="float32")
# outputs_test = pd.read_csv('C:/Users/mark_/userdata/Output/y_test_mm_ns.csv', sep=';', decimal='.', header=None, index_col=False).to_numpy()

# nrep = 100
# # Y_train_rep = []
# # Y_v_rep = []
# Yhat_rep_lstm = []
# Yhat_v_rep_lstm = []

# # b = [4, 16, 32]  # b = batch size

# for i in range(nrep):
   
#     #########  Use only during loo runs
#     inputs_train_loo = inputs_train
#     outputs_train_loo = outputs_train
#     rdel = np.random.randint(0, high=len(inputs_train_loo)+1)# rdel = random selection of the column to be deleted
     
#     inputs_train_loo = np.delete(inputs_train_loo, rdel, axis=0).reshape(-1,4) # New array with loo column selected at random
#     outputs_train_loo = np.delete(outputs_train_loo, rdel, axis=0).reshape(-1,1) # New array with loo column selected at random
    
#     # index = np.array(np.delete(keys, rdel, axis=None))
#     # X_loo = [L9[x+1] for x in index] 
#     # # X_data_mm = np.vstack(X_loo)
#     # X_data_mm = np.concatenate((X_loo), axis=0).astype('float') # New array with loo selected at random
#     #########  Use only during loo runs
    
    
#     Yhat, Yhat_v = AFBR_lstm(inputs_train_loo, outputs_train_loo, inputs_test, outputs_test)  # , b[i] 
    
#     Yhat_rep_lstm.append(Yhat)  # training replicates
#     Yhat_v_rep_lstm.append(Yhat_v)   # validation replicates
    

# np.savetxt('C:/Users/mark_/userdata/Output/Yhat_k_lstm_mm_100reps_loo.csv',np.asarray(Yhat_rep_lstm).reshape(-1,100), delimiter=";", fmt="%10.4f") # save the results obtained using the keras lstm, not shuffled data
# np.savetxt('C:/Users/mark_/userdata/Output/Yhat_v_k_lstm_mm_100reps_loo.csv',np.asarray(Yhat_v_rep_lstm).reshape(-1,100), delimiter=";", fmt="%10.4f") # save the results obtained using the keras lstm,not shuffled data


#Save for single runs
# np.savetxt('C:/Users/mark_/userdata/Output/y_test_k_lstm_mm.csv',outputs_test, delimiter=";", fmt="%10.4f") # save the results obtained using the keras lstm, not shuffled data
# np.savetxt('C:/Users/mark_/userdata/Output/Yhat_k_lstm_mm.csv',Yhat, delimiter=";", fmt="%10.4f") # save the results obtained using the keras lstm, not shuffled data
# np.savetxt('C:/Users/mark_/userdata/Output/Yhat_v_k_lstm_mm.csv',Yhat_v, delimiter=";", fmt="%10.4f") # save the results obtained using the keras lstm,not shuffled data

################ Replicated MLP model building  ######################

# # # save results of replicate runs to the PC for subsequent descriptive statistics
# # # np.savetxt("C:/Users/mark_/userdata/Output/Yhat_test_all.csv", Yhat_test_all, delimiter=',')
# # # np.savetxt("C:/Users/mark_/userdata/Output/Yhat_v_mlp_4b.csv", Yhat_test_all, delimiter=',')
# # # np.savetxt("C:/Users/mark_/userdata/Output/Yhat_v_mlp_4c.csv", Yhat_test_all, delimiter=',')
# # np.savetxt("C:/Users/mark_/userdata/Output/Yhat_v_mlp_12-l.csv", Yhat_test_all, delimiter=',')

# # # save results of replicate runs to the CLUSTERS for subsequent descriptive statistics
# # np.savetxt("/scratch/mmccormi1/X_train_mm.csv", X_train_mm, delimiter=',')
# # np.savetxt("/scratch/mmccormi1/X_test_mm.csv", X_test_mm, delimiter=',')
# # np.savetxt("/scratch/mmccormi1/Y_train_all.csv", Y_train_all, delimiter=',')
# # np.savetxt("/scratch/mmccormi1/Y_test_mm.csv", y_test_mm, delimiter=',')
# np.savetxt("/scratch/mmccormi1/Yhat_train_all_h.csv", Yhat_train_all, delimiter=',')  

# # np.savetxt("/scratch/mmccormi1/Yhat_v_mlp_4b.csv", Yhat_test_all, delimiter=',')
# # np.savetxt("/scratch/mmccormi1/Yhat_test_all_b.csv", Yhat_test_all, delimiter=',') # b, shuffled data, 1-layer mlp, 6 x nodes version

# # np.savetxt("/scratch/mmccormi1/Yhat_test_all_c.csv", Yhat_test_all, delimiter=',')   # c, not shuffled data, 6-layer mlp
# # np.savetxt("/scratch/mmccormi1/Yhat_test_all_d.csv", Yhat_test_all, delimiter=',')  # d, shuffled data, 6-layer mlp
# # np.savetxt("/scratch/mmccormi1/Yhat_test_all_e.csv", Yhat_test_all, delimiter=',')  # e, shuffled data, 3-layer mlp
# # np.savetxt("/scratch/mmccormi1/Yhat_test_all_f.csv", Yhat_test_all, delimiter=',') # f, shuffled data, 2-layer mlp
# # np.savetxt("/scratch/mmccormi1/Yhat_test_all_g.csv", Yhat_test_all, delimiter=',') # g, shuffled data, 12-layer mlp
# np.savetxt("/scratch/mmccormi1/Yhat_test_all_h.csv", Yhat_test_all, delimiter=',') # h, shuffled data, 1-layer mlp, 12 x nodes version
# # np.savetxt("/scratch/mmccormi1/Yhat_v_mlp_4h.csv", Yhat_test_all, delimiter=',')

# # np.savetxt("/scratch/mmccormi1/Yhat_v_mlp_4d_do5.csv", Yhat_test_all, delimiter=',')


# # # np.savetxt("C:/Users/mark_/Documents/02_Professional/03_Unil/SATW/MDPI_paper/MDPI_article/Results/y_train_mm", y_train_mm, delimiter=',')
# # np.savetxt("C:/Users/mark_/Documents/02_Professional/03_Unil/SATW/MDPI_paper/MDPI_article/Results/y_test_mm_91-100", y_test_mm, delimiter=',')

# # N_coor = np.corrcoef(Yhat_test_all, Yhat_v, rowvar=False)
# # # corr, p_value = scipy.stats.pearsonr(Yhat_test_all, Yhat_v)
# # print('Pearsons correlation: %.3f' % N_coor)
# # print('Pearsons p-value: %.3f' % p_value)

# # #### Concatenate replicates ####
# # # Yhat_test_all_H = np.loadtxt("C:/Users/mark_/Documents/02_Professional/03_Unil/SATW/MDPI_paper/MDPI_article/Data/Yhat_test_all_A", delimiter=',')
# # # Yhat_test_all_I = np.loadtxt("C:/Users/mark_/Documents/02_Professional/03_Unil/SATW/MDPI_paper/MDPI_article/Data/Yhat_test_all_B", delimiter=',')
# # # Yhat_test_all_J = np.loadtxt("C:/Users/mark_/Documents/02_Professional/03_Unil/SATW/MDPI_paper/MDPI_article/Data/Yhat_test_all_C", delimiter=',')
# # # Yhat_test_all_K = np.loadtxt("C:/Users/mark_/Documents/02_Professional/03_Unil/SATW/MDPI_paper/MDPI_article/Data/Yhat_test_all_D", delimiter=',')
# # yhat_regression_loo_27 = np.loadtxt("C:/Users/mark_/Documents/02_Professional/03_Unil/SATW/MDPI_paper/MDPI_article/Data/yhat_regression_loo_27", delimiter=',')
# # yhat_regression_loo_23 = np.loadtxt("C:/Users/mark_/Documents/02_Professional/03_Unil/SATW/MDPI_paper/MDPI_article/Data/yhat_regression_loo_23", delimiter=',')
# # # Yhat_test_all_G = np.loadtxt("C:/Users/mark_/Documents/02_Professional/03_Unil/SATW/MDPI_paper/MDPI_article/Data/Yhat_test_all_G", delimiter=',')

# # yhat_regression_loo_50 =np.concatenate([yhat_regression_loo_27, yhat_regression_loo_23], axis=0) #, Yhat_test_all_E, Yhat_test_all_F, Yhat_test_all_G
# # Transpose = np.transpose(yhat_regression_loo_50, axes=None)
# # np.savetxt("C:/Users/mark_/Documents/02_Professional/03_Unil/SATW/MDPI_paper/MDPI_article/Data/yhat_regression_loo_50", Transpose, delimiter=',')

# # # Summary = pd.DataFrame(data = Yhat_test_all_23)
# # # Summary.describe()
# # # for m in range(23):
# # #     print(Summary.T[m].mean())
# # #     print(Summary.T[m].std())
# # #     print(100*Summary.T[m].std()/Summary.T[m].mean())





# ############### Multiple simulations   ##################
# ###  Reload and use saved model to run simulations of new influent flow profiles 
# Exp_data_CVred = pd.read_excel('C:\\Users\\mark_\\Anaconda3\Data\SATW_project.xlsx', sheet_name='CVred') 
# Exp_data_inflow = pd.read_excel('C:\\Users\\mark_\\Anaconda3\Data\SATW_project.xlsx', sheet_name='import') 

# Exp_data_CVred = pd.read_excel('C:\\Users\\mark_\\Anaconda3\Data\SATW_project.xlsx', sheet_name='CVred') 
# Exp_data_inflow = pd.read_excel('C:\\Users\\mark_\\Anaconda3\Data\SATW_project.xlsx', sheet_name='import') 
#     # Start date = 11 May. End dtae = 15 December. No of data points (days) = 170.
# DCVr = Exp_data_CVred.iloc[:,:2].set_index('Date_', inplace=False) # derived experimental CV reduction (kJ/l) using equation 5. Exp_data_CVred.iloc[:,:2].set_index('Date_', inplace=False)
# DCVo = Exp_data_CVred.iloc[:,3:5].set_index('Date_Exp', inplace=False) # original experimental CV reduction (kJ/l) using equation 5
# # Date_ = Exp_data_CVred.iloc[:,5]
# Merge_exp = DCVr.join(DCVo)
    
########   Import experimental influent flow rate and select simulated flow rate (l/day)  ###########
# Qr = Exp_data_inflow.iloc[:,[3]]#.set_index('Date_', inplace=False) # 
# Qr = pd.read_excel('C:\\Users\\mark_\\Anaconda3\Data\SATW_project.xlsx', sheet_name='Experiment_L9_V3').iloc[:170,[7]] 
# plt.plot(Qr)
# ### Creat simulation data one time.
# from satw_data import NN_simulationdata
# AFBR_s = keras.models.load_model('C:/Users/mark_/Anaconda3/envs/Tensorflow_v2/Mark/AFBR_mlp_v2-3-layer.h5')
# # AFBR_s = keras.models.load_model('C:/Users/mark_/Anaconda3/envs/Tensorflow_v2/Mark/AFBR_1.h5')
# # AFBR_s.summary()  # summarize model.
# ns = len(Qr)  # Experimental samples + derived samples = 170 samples total. Merge_exp
# SX_test_mm = NN_simulationdata(ns, Qr)

# np.min(Qr)

###############################################################################
########### Use previously created MLP model to run multiple simulations
from satw_data_p import NN_simulationdata
from satw_nn_p import AFBR_MLP_sim

Qr = IFR_ref_all
ns = len(Qr)  # number of samples per experiment (170)

perm_5 = list(itertools.product([4,8,12,24,36],[1,1.85,2.7,6.85,11],[0.5,1.15,1.8,2.9,4])) # sphere diameter, Material activity, H/D
#print('perm_5',perm_5)
print('Length perm_5', len(perm_5))
        
Rep = []#Array of Yhat of the replicates.
dSQ_test = []  #Array of Q influent of the replicates.
# Rep2 = []
exp =np.arange(1,126,1)  # Experiment number
nsim =  20# Enter the number of simulations to run. 20 in the MDPI article
##for i in range(len(exp)):

######### Create mock data once and then reload
##SX_test_mm = NN_simulationdata(perm_5) #, Q_Ex_sim_mm 
##print('SX_test_mm shape',SX_test_mm.shape)
##np.savetxt("C:/Users/mark_/mark_data/Output/SX_test_mm_127_516.csv", SX_test_mm, delimiter=",", fmt="%10.5f")
SX_test_mm = pd.read_csv('C:/Users/mark_/mark_data/Output/SX_test_mm_127_516.csv', sep=',', decimal='.', header=None, index_col=False)
##
##np.savetxt("C:/Users/mark_/mark_data/Output/SX_test_mm_127_321.csv", SX_test_mm, delimiter=",", fmt="%10.5f")
##SX_test_mm = pd.read_csv('C:/Users/mark_/mark_data/Output/SX_test_mm_127_321.csv', sep=',', decimal='.', header=None, index_col=False)
##
##np.savetxt("C:/Users/mark_/mark_data/Output/SX_test_mm_322_516.csv", SX_test_mm, delimiter=",", fmt="%10.5f")
##SX_test_mm = pd.read_csv('C:/Users/mark_/mark_data/Output/SX_test_mm_322_516.csv', sep=',', decimal='.', header=None, index_col=False)

print('SX_test head', SX_test_mm.head(n=200))

####NN_simulationdata(perm_5) 
for i in range(nsim):   # 
    #SX_test_mm, Q_Ex_sim_mm = NN_simulationdata(perm_5)  #SX_test_mm = ns, Qr generates a simulation data from a new randomly generated influent flow profile. 
    #print('SX_test_mm, simulation data',SX_test_mm)
    ##SX_test_mm = MinMax scaled complete simulation test dataset (mech + inf flow). new ns = number of samples, Q_Ex_Sim = new randomly generated influent flow rate generated for each simulation run.
    Rep.append(AFBR_MLP_sim(SX_test_mm))#, AFBR_s)) #  Runs the simulation using the new data and appends the result to a file of all simulation replicates.
    # dSQ_test.append(SX_test_mm[:,[3]])
    #dSQ_test.append(np.asarray(Q_Ex_sim_mm))   ##
##Rep = AFBR_MLP_sim(SX_test_mm)     
##dSQ_t = np.asarray(dSQ_test)
##dtest = np.hstack(dSQ_test)
## np.savetxt("C:/Users/mark_/userdata/Output/dtest.csv", dtest, fmt='%.4g', delimiter=',') #%.4g
##dtestr = np.asarray([x/10 for x in dtest])   #????
##dSQ_mean = dtest.mean(axis = 0)
##dSQ_mean = dtest.mean(axis = 1)   # The mean over all replicates of Qinfluent for each day of the data
## dSQ_mean = np.asarray(dSQ_test).mean(axis = 0)
##
## dSQ_std = dtest.std(axis = 0)
##dSQ_std = dtest.std(axis = 1) # The standard deviation over all replicates of Qinfluent for each day of the data
##dSQ_std = np.asarray(dSQ_std).std(axis = 0)
###### Compute stats and make graphs of the multiple simulation results   ################3

# Rep.to_csv('C:/Users/mark_/Anaconda3/envs/Tensorflow_v1/Mark/bg_sum_mlp.csv', sep=';', index=False, decimal=',')
# 
# print('Samples, Exp', len(Exp)) #,'Replicate matrix shape', Rep.shape
#Rep= np.asarray(Rep)  #  Array of the results of simulation (cumulative calorific value reduction)
# print((np.mean(Rep, axis=1)))
# print((np.mean(Rep, axis=0)).argsort()[:126])  #negative causes sort in descending order. L9 -> 28. L3^5 -> 126
##y_data_mm = scale_tanh.fit_transform(y_data)
##scaler = MinMaxScaler(feature_range=(0,1), copy=True)
##Rep = scaler.inverse_transform(Rep)
print('Rep - length of all simulation replicate predictions',len(Rep))
Rep = np.asarray(Rep)
print('shape Rep array, before shaping', np.shape(Rep))
print('Head 1-50 Rep array, before shaping', Rep[:50])


Repdf_t = np.mean(Rep,axis=0)
print('shape Repdf_t mean array, before shaping', np.shape(Repdf_t))
print('Repdf_t mean array, before shaping', Repdf_t)
##Rep = Rep.flatten()
##print('shape Flattened Rep', np.shape(Rep))
##Rep_C = Rep.reshape(-1,21250, order='C') #(-1,1) 
##print('Reshaped Rep, C  order (no sim x 21250)', np.shape(Rep_C))
##Rep_rs = Rep.reshape(-1,len(perm_5), order='F') #(-1,1) 
##print('Reshaped Rep, F order (no sim x 21250)', np.shape(Rep_rs))
Rep_rs = Repdf_t.reshape(-1,len(perm_5), order='F') #(-1,1) 
print('Reshaped Repdf_t, F order (no sim x 21250)', np.shape(Rep_rs))
Rep_F = pd.DataFrame(Rep_rs)
#Rep = Rep.mean(axis=0)


##low = np.argwhere(Rep_rs < 0.3)
##print('low',len(low))

##plt.figure()
###plt.plot(Rep[:30])
##Rep_F.iloc[:50,:100].plot()
##plt.title('Rep_df. predictions after reshaping')
##plt.show()
#Repdf = pd.DataFrame(Rep) 
# np.savetxt("C:/Users/mark_/userdata/Output/Simulation_replicates.csv", Repdf, fmt='%.4f', delimiter=',') #%.4g
print('Simulation Repdf describe',Rep_F.describe()) 
# np.savetxt("C:/Users/mark_/userdata/Output/Simulation_replicates_stats.csv", Repdf.describe(), fmt='%.4f', delimiter=',') 
#Sim_stats = Repdf.describe()


### Plot of mean and standard deviation of the influent flow rate over the replicates
### x_pos = np.arange(len(dtest))#)[0]
### Qmean = dSQ_mean
### Qstd = dSQ_std
### # Build the plot
### fig, ax = plt.subplots(figsize=(12,12))
### ax.bar(x_pos, Qmean, yerr=Qstd, align='center', alpha=0.5, ecolor='black', capsize=10)
### ax.set_ylabel('Influent flow rate ($liters.day^{-1}$)')
### ax.set_xticks(x_pos)
### ax.set_xticklabels(dtest)
### ax.set_title('Mean influent flow rate ($liters.day^{-1}$) used for simulations')
### # ax.yaxis.grid(True)
### plt.tight_layout()
### plt.show()
###  plt.savefig('C:/Users/mark_/Documents/02_Professional/03_Unil/SATW/MDPI_paper/MDPI_article/MDPI_article_v21/figures/F8_inf_bar.svg', format="svg")
##
##plt.plot(Repdf.mean())
### # # print(Repdf.mean())
### # print(Repdf.std().max())
### # print(Repdf.std()[123])
##RepA = Rep.mean(axis=1)
##print('A, mean of replicates',RepA)
##RepB = RepA.to_numpy()
##print('B, shape',np.shape(RepB))
##print('B, mean of replicates',RepB)
##RepC = RepB.reshape(ns,len(perm_5), order='C')  # Order = F => count 170 samples (rows), then make new column. 170 x 125
##print('C shape',np.shape(RepC))
##D = C.transpose()
##print('D',D)

Repdf_t = Rep_F.mean(axis=0)  # Axis = 0 => take the mean along the row (=> of the column).
Repdf_t = Rep_F.sum(axis=0)   # Axis = 0 => take the sum along the row (=> of the column).

Repdf_t = Rep_F 
plt.figure()
Repdf_t.plot()
plt.title('Repdf_t, sum/mean simulation CVred')
plt.show()

####Repdf_t = Rep.mean(axis=1).to_numpy.reshape(len(ns),len(perm_5)).transpose()#Lines = 170 samples. Columns = 125 simulated experiments. 
###print('Repdf_t', Repdf_t)
Rank = Repdf_t.rank(ascending=False )  # Ascending = False means that row 1 has the highest ranking CV reduction. ,
print('Rank order, high to low',Rank)

print('List position of Rank[0]', Rank[0])

print("Order is: ESD, MAT, HDR")
first = np.where(Rank == 1)
print('First',perm_5[first[0].item(0)])

second = np.where(Rank == 2)
print('Second',perm_5[second[0].item(0)])

third = np.where(Rank == 3)
print('Third',perm_5[third[0].item(0)])

fourth= np.where(Rank == 4)
print('Fourth',perm_5[fourth[0].item(0)])

fifth= np.where(Rank == 5)
print('Fifth', perm_5[fifth[0].item(0)])

####print('Mech parameter values of test rank upper limit for test of significant difference with top ranked:',perm_5[test_rank[0].item(0)])
####
####print('Test rank "other" upper limit position:' + str(test_rank))
####
####print('perm_5 test, third', perm_5[2])
####
####print('where type', np.where(Rank == 3)[0])
####
####print('Mech parameter values of rank 1',perm_5[np.where(Rank == 1)[0][0]]) 
##
##print('The ranking of all 125 simulated experiments:') 
##for i in range(1,126):
##    print(perm_5[np.where(Rank == i)[0][0]])
##    


####
##### # print('Number of simulation replicates, Rep', len(Rep))
####
##### ### t-test ### to show difference between highest ranking simulated experiment results
##### ## A test for the null hypothesis that 2 independent samples have identical average (expected) values. 
####### ## If the p-value is smaller than the threshold, e.g. 1%, 5% or 10%, then we reject the null hypothesis of equal averages.
####print('Mean of first ranked',Repdf_t.iloc[int(first[0])].mean())
####print('Std of first ranked',Repdf_t.iloc[int(first[0])].std())
####print('Mean of second ranked',Repdf_t.iloc[int(second[0])].mean())
####print('Std of second ranked',Repdf_t.iloc[int(second[0])].std())
####print('Mean of third ranked',Repdf_t.iloc[int(third[0])].mean())
####print('Std of third ranked',Repdf_t.iloc[int(third[0])].std())
####
####
##print('Rep_rs[0]',Rep_rs[0])
##print('int(first[0]',int(first[0]))
##print('int(second[0]',int(second[0]))
##print('Rep_rs',Rep_rs)
##print('Rep_rs[:,int(first[0])]',Rep_rs[:,int(first[0])])
##print('Rep_rs[:,int(second[0])]',Rep_rs[:,int(second[0])])
##print('Rep_rs[:,int(first[0]), shape]',Rep_rs[:,int(first[0])].shape)
##print('Rep_rs[:,int(second[0])]',Rep_rs[:,int(second[0])].shape)
###A = np.asarray(Rep_rs[0][int(first[0])])
##A = Rep_rs[:,int(first[0])]
##print('A',A)
##print('A[:]',A[:])
##B = Rep_rs[:,int(second[0])]
##print('B',B)

##np.seterr(invalid='ignore')
###np.seterr(divide='ignore', invalid='ignore')
##statistic_12, pvalue_12 = scipy.stats.ttest_ind(a=Rep_rs[:,int(first[0])],b=Rep_rs[:,int(second[0])])# axis=None, equal_var=False, nan_policy='propagate', alternative = 'two-sided')
####statistic_12, pvalue_12 = scipy.stats.ttest_ind(Rep_rs[0][int(first[0])], Rep_rs[0][int(second[0])], axis=0, equal_var=False, nan_policy='propagate', alternative = 'greater')
###statistic_12, pvalue_12 = scipy.stats.ttest_ind(Rep.transpose()[int(first[0])], Rep.transpose()[int(second[0])], axis=0, equal_var=False, nan_policy='propagate', alternative = 'greater')  # This worked. Keep it.
### pvalue = scipy.stats.ttest_ind(Repdf.transpose()[int(first[0])], Repdf.transpose()[int(second[0])], axis=0, equal_var=False, nan_policy='propagate', alternative = 'greater') # greater => the average of one sample (a) is greater than the average of the other sample (b). axis 0 represents rows
##print('t-test statistic, first-second', statistic_12)  # The null hypothesis of the t-test is no difference in means.
##print('t-test p-value, first-second', pvalue_12)  # The p-value is the percentage of the t-distribution histogram that is more extreme. Often what we say that there is a significant difference if the p-value is less than .05, i.e. the percent of samples that would be more extreme is less than .05.
######
##statistic_13, pvalue_13 = scipy.stats.ttest_ind(a=Rep_rs[:,int(first[0])],b=Rep_rs[:,int(third[0])])
### pvalue = scipy.stats.ttest_ind(Repdf.transpose()[int(first[0])], Repdf.transpose()[int(second[0])], axis=0, equal_var=False, nan_policy='propagate', alternative = 'greater') # greater => the average of one sample (a) is greater than the average of the other sample (b). axis 0 represents rows
##print('t-test statistic, first-third', statistic_13)  # The null hypothesis of the t-test is no difference in means.
##print('t-test p-value, first-third', pvalue_13)  # The p-value is the percentage of the t-distribution histogram that is more extreme. Often what we say that there is a significant difference if the p-value is less than .05, i.e. the percent of samples that would be more extreme is less than .05.
##
##statistic_14, pvalue_14 = scipy.stats.ttest_ind(a=Rep_rs[:,int(first[0])],b=Rep_rs[:,int(fourth[0])])
### pvalue = scipy.stats.ttest_ind(Repdf.transpose()[int(first[0])], Repdf.transpose()[int(second[0])], axis=0, equal_var=False, nan_policy='propagate', alternative = 'greater') # greater => the average of one sample (a) is greater than the average of the other sample (b). axis 0 represents rows
##print('t-test statistic, first-fourth', statistic_14)  # The null hypothesis of the t-test is no difference in means.
##print('t-test p-value, first-fourth', pvalue_14)  # The p-value is the percentage of the t-distribution histogram that is more extreme. Often what we say that there is a significant difference if the p-value is less than .05, i.e. the percent of samples that would be more extreme is less than .05.
##
##statistic_15, pvalue_15 = scipy.stats.ttest_ind(Rep.transpose()[int(first[0])], Rep.transpose()[int(fifth[0])], axis=0, equal_var=False, nan_policy='propagate', alternative = 'greater')
### pvalue = scipy.stats.ttest_ind(Repdf.transpose()[int(first[0])], Repdf.transpose()[int(second[0])], axis=0, equal_var=False, nan_policy='propagate', alternative = 'greater') # greater => the average of one sample (a) is greater than the average of the other sample (b). axis 0 represents rows
##print('t-test statistic, first-fifth', statistic_15)  # The null hypothesis of the t-test is no difference in means.
##print('t-test p-value, first-fifth', pvalue_15)  # The p-value is the percentage of the t-distribution histogram that is more extreme. Often what we say that there is a significant difference if the p-value is less than .05, i.e. the percent of samples that would be more extreme is less than .05.

##
##statistic, pvalue = scipy.stats.ttest_ind(Rep.transpose()[int(first[0])], Rep.transpose()[int(test_rank[0])], axis=0, equal_var=False, nan_policy='propagate', alternative = 'greater')
### pvalue = scipy.stats.ttest_ind(Repdf.transpose()[int(first[0])], Repdf.transpose()[int(second[0])], axis=0, equal_var=False, nan_policy='propagate', alternative = 'greater') # greater => the average of one sample (a) is greater than the average of the other sample (b). axis 0 represents rows
##print('t-test statistic, first-other', statistic)  # The null hypothesis of the t-test is no difference in means.
##print('t-test p-value', pvalue)  # The p-value is the percentage of the t-distribution histogram that is more extreme. Often what we say that there is a significant difference if the p-value is less than .05, i.e. the percent of samples that would be more extreme is less than .05.
##
##statistic, pvalue = scipy.stats.ttest_ind(Rep.transpose()[int(second[0])], Rep.transpose()[int(third[0])], axis=0, equal_var=False, nan_policy='propagate', alternative = 'greater')
### pvalue = scipy.stats.ttest_ind(Repdf.transpose()[int(first[0])], Repdf.transpose()[int(second[0])], axis=0, equal_var=False, nan_policy='propagate', alternative = 'greater') # greater => the average of one sample (a) is greater than the average of the other sample (b). axis 0 represents rows
##print('t-test statistic, second-third', statistic)
##print('t-test p-value', pvalue)
# # print(Rep.transpose()[11])
# # print(Rep.transpose()[1])
# # # Repsum = Repdf.sum(axis=0)
# # Repsumdf = pd.DataFrame(Repsum)
# # print(Repsumdf.describe()) 
# # print(Repsumdf.rank(axis=0, ascending=False))

# # scaler = MinMaxScaler(feature_range=(0,1), copy=True)
# # Rep_mm= scaler.fit_transform(Rep)#.astype(np.float32)

# ##########  Calculate bin counts from the mean CV reduction of the replicated simulations    ##########
##bins = 17  #  Number of histogram bins = 17 (170/10 = 17)
##bins = np.linspace(0, 17, 17)
scaler = MinMaxScaler(feature_range=(0,1), copy=True)
bins_ye = np.linspace(127, 516, 17).reshape(-1, 1)#   # Bin limits. Experiment range: 127, 516, 17.  Low range: 127, 150, 17
bins_y = scaler.fit_transform(bins_ye)#.astype(int)  # 
Qr = scaler.fit_transform(Qr)#.reshape(-1, 1)#.astype(int)  Qr = IFR_ref_all = experimental influent flow rate
#Q_flat = scaler.fit_transform(SX_test_mm.iloc[:,3].to_numpy().reshape(-1, 1))#[i for i in testlist]
Q_flat = SX_test_mm.iloc[:,3].to_numpy().reshape(-1, 1)#[i for i in testlist]  MinMax scaled complete simulation influent flow rate test dataset. 170 x 125 = 21250 days

##print('bins, using lin space', bins)
print('bins y', bins_y)
print('bins y [3]', bins_y[3][0])
print('Qr', Qr)
print('Qr [7]', Qr[7][0])

#Q_flat= dtest.flatten()
##testlist = [SX_test_mm.iloc[:,3]]

print('Q_flat', np.shape(Q_flat))
print('Q_flat head', Q_flat)
print('Q_flat [9,0]', Q_flat[9])

B_1 = []; B_2 = []; B_3 = []; B_4 = []; B_5 = []; B_6 = []; B_7 = []; B_8 = [];B_9 = [];
B_10 = []; B_11 = []; B_12 = []; B_13 = []; B_14 = []; B_15 = []; B_16 = []; B_17=[];

B_1c = 0; B_2c = 0; B_3c = 0; B_4c = 0; B_5c = 0; B_6c = 0; B_7c = 0; B_8c = 0;B_9c = 0;
B_10c = 0; B_11c = 0; B_12c = 0; B_13c = 0; B_14c = 0; B_15c = 0; B_16c = 0; B_17c=0;

for i in range(len(Q_flat)):
    if (Q_flat[i][0]>bins_y[0]) and (Q_flat[i][0]<bins_y[1]):
        B_1.append(Q_flat[i])
        B_1c+=1
    if (Q_flat[i][0]>bins_y[1]) and (Q_flat[i][0]<bins_y[2]):
        B_2.append(Q_flat[i])
        B_2c+=1
    if (Q_flat[i][0]>bins_y[2]) and (Q_flat[i][0]<bins_y[3]):
        B_3.append(Q_flat[i])
        B_3c+=1
    if (Q_flat[i][0]>bins_y[3]) and (Q_flat[i][0]<bins_y[4]):
        B_4.append(Q_flat[i])
        B_4c+=1
    if (Q_flat[i][0]>bins_y[4]) and (Q_flat[i][0]<bins_y[5]):
        B_5.append(Q_flat[i])
        B_5c+=1
    if (Q_flat[i][0]>bins_y[5]) and (Q_flat[i][0]<bins_y[6]):
        B_6.append(Q_flat[i])
        B_6c+=1
    if (Q_flat[i][0]>bins_y[6]) and (Q_flat[i][0]<bins_y[7]):
        B_7.append(Q_flat[i])
        B_7c+=1
    if (Q_flat[i][0]>bins_y[7]) and (Q_flat[i][0]<bins_y[8]):
        B_8.append(Q_flat[i])
        B_8c+=1
    if (Q_flat[i][0]>bins_y[8]) and (Q_flat[i][0]<bins_y[9]):
        B_9.append(Q_flat[i])
        B_9c+=1
    if (Q_flat[i][0]>bins_y[9]) and (Q_flat[i][0]<bins_y[10]):
        B_10.append(Q_flat[i])
        B_10c+=1
    if (Q_flat[i][0]>bins_y[10]) and (Q_flat[i][0]<bins_y[11]):
        B_11.append(Q_flat[i])
        B_11c+=1
    if (Q_flat[i][0]>bins_y[11]) and (Q_flat[i][0]<bins_y[12]):
        B_12.append(Q_flat[i])
        B_12c+=1
    if (Q_flat[i][0]>bins_y[12]) and (Q_flat[i][0]<bins_y[13]):
        B_13.append(Q_flat[i])
        B_13c+=1
    if (Q_flat[i][0]>bins_y[13]) and (Q_flat[i][0]<bins_y[14]):
        B_14.append(Q_flat[i])
        B_14c+=1
    if (Q_flat[i][0]>bins_y[14]) and (Q_flat[i][0]<bins_y[15]):
        B_15.append(Q_flat[i])
        B_15c+=1
    if (Q_flat[i][0]>bins_y[15]) and (Q_flat[i][0]<bins_y[16]):
        B_16.append(Q_flat[i])
        B_16c+=1
##    else:
##        B_17.append(Q_flat[i])
##        B_17c+=1
    if (Q_flat[i][0]>bins_y[16]) and (Q_flat[i][0]<bins_y[17]):
        B_17.append(Q_flat[i])
        B_17c+=1

binss =  bins_ye.flatten()#np.linspace(0, 16, 16) # Flattened bin limits. [bins_y]
bins = [B_1, B_2, B_3, B_4, B_5, B_6, B_7, B_8, B_9, B_10, B_11, B_12, B_13, B_14, B_15, B_16, B_17]
##bin_mean = [np.mean(bins[c]) for c in range(len(bins))]
#bin_count = [np.count_nonzero(bins[c]) for c in range(len(bins))]
bin_count = [B_1c, B_2c, B_3c, B_4c, B_5c, B_6c, B_7c, B_8c, B_9c, B_10c, B_11c, B_12c, B_13c, B_14c, B_15c, B_16c, B_17c]
##bin_std = [np.std(bins[c]) for c in range(len(bins))]
##cvar= np.divide(np.multiply(bin_std,2), bin_mean)
        
# plt.figure(figsize=(12,12))
# fig, (ax1) = plt.subplots(nrows=1)
# ax1.bar(binss, bin_count)
# ax2 = ax1.twinx()
# ax2.bar(binss, bin_std, width=0.1, align= 'center', color = "red")
# plt.show()
# plt.bar(binss, bin_count, yerr=np.divide(bin_std,bin_mean), align= 'edge',width=0.5)  #, tick_label=bins_str
# plt.bar(bins, counts_dSQ_r, align= 'edge', width=-0.5, tick_label=bins_str)
# plt.hist(dtest, bins=bins, density=False, histtype='step', cumulative = True)

counts, bins_h = np.histogram(Qr, range = (0,1), bins = 17)  # data from mock experiments Qr = MinMax scaled influent flow rate from real experiments. range = (127,516)
counts_s, bins_s = np.histogram(Q_flat, range = (0,1), bins = 17)  # data from mock experiments Qr = MinMax scaled influent flow rate from real experiments. range = (127,516)
print('counts_s, Simulation bin counts',counts_s)
print('bins_ye, Experiment bin limits',bins_ye)
print('binss, flattened bin limits',binss)
print('bins_h, flattened bin limits returned from the histogram function',bins_h)
print('bins, List of the values contained in each of the bins',bins)
print('counts, bin counts reference experiment', counts)
print('bin_count, bin counts simulated experiment',bin_count)
print('B_16c, bin content',B_16c)
print('B_17c, bin content',B_17c)
print('B_16, bin content',B_16)
print('B_17, bin content',B_17)
print('Q_flat[16], bin content',Q_flat[16])
print('Q_flat[16][0], bin content',Q_flat[16][0])
fig, ax1 = plt.subplots(figsize=(16.2,10))
label = ["Mock experiment", "Simulations (Mean)"]
#plt.ylim([0,60])
ax1.bar(binss, counts, width=-10, align= 'edge', color = "blue" ) #  The real experiment.
#ax1.ylim([0,60])  #np.multiply(np.divide(counts,nsim), 1)
# ax2 = ax1.twinx()
#ax1.bar(binss, np.divide(bin_count,125), width=10, align= 'edge', color = "green")#  The mean of the 125 simulated experiments.
ax1.bar(binss, np.divide(counts_s,125), width=10, align= 'edge', color = "green")#  The mean of the 125 simulated experiments.
ax1.tick_params(axis="y", labelsize=16, direction="in")
ax1.tick_params(axis="x", labelsize=16, direction="out")
plt.legend(label)
plt.xlabel('Influent flow rate [l/day]', fontsize=24)
plt.ylabel('Bin count [days]', fontsize=24)

#plt.title('Distribution of Mock experiments and Simulation (mean) influent flow rates', fontsize='large')
plt.savefig('C:/Users/mark_/Documents/02_Professional/03_Unil/SATW/MDPI_paper/Journal communication/Sustainability/Round two special issue/Figures_originals/Fig_Qinf_binned.svg', format="svg", bbox_inches="tight", dpi=300)
plt.savefig('C:/Users/mark_/Documents/02_Professional/03_Unil/SATW/MDPI_paper/Journal communication/Sustainability/Round two special issue/Figures_originals/Fig_Qinf_binned.pdf', format="pdf", bbox_inches="tight", dpi=300)
plt.show()


#### Test
##fig, (ax1, ax2) = plt.subplots(figsize=(16.2,10))
###label = ["Mock experiment", "Simulations (Mean)"]
### plt.ylim([0,60])
##ax1.bar(binss, counts, width=-10, align= 'edge', color = "blue" ,label = "Mock experiment") #  The real experiment.
### ax1.ylim([0,60])  #np.multiply(np.divide(counts,nsim), 1)
### ax2 = ax1.twinx()
##ax2.bar(binss, np.divide(bin_count,nsim*len(exp)), width=10, align= 'edge', color = "green", label = "Simulations (Mean)")#  The mean of the simulated experiments.
##plt.legend(loc="upper right")
##plt.xlabel('Influent flow rate [l/day]', fontsize=24)
##plt.ylabel('Bin count [days]', fontsize=24)
##
###plt.title('Distribution of Mock experiments and Simulation (mean) influent flow rates', fontsize='large')
####plt.savefig('C:/Users/mark_/Documents/02_Professional/03_Unil/SATW/MDPI_paper/Journal communication/Sustainability/Round two special issue/Figures_originals/Fig_Qinf_binned.svg', format="svg", bbox_inches="tight", dpi=300)
####plt.savefig('C:/Users/mark_/Documents/02_Professional/03_Unil/SATW/MDPI_paper/Journal communication/Sustainability/Round two special issue/Figures_originals/Fig_Qinf_binned.pdf', format="pdf", bbox_inches="tight", dpi=300)
##plt.show()
 
##fig, axs = plt.subplots(figsize=(16.2,10))
####    plt.plot(history.history['loss'], c="g", linestyle="-", linewidth = 4, label='training_loss') #, 'r'train_history
####    plt.plot(history.history['val_loss'], c="k", linestyle="-", linewidth = 2, label='test_loss') #, 'b'val_history, val_loss
##    axs.plot(history.history['loss'], c="g", linestyle="-", linewidth = 4, label='training_loss') #, 'r'train_history
##    axs.plot(history.history['val_loss'], c="k", linestyle="-", linewidth = 2, label='test_loss') #, 'b'val_history, val_loss
##    axs.tick_params(axis="y", labelsize=16, direction="in")
##    axs.set_ylabel("Mean Squared Error", fontsize=24)
##    axs.tick_params(axis="x", labelsize=16, direction="in")
##    axs.set_xlabel("Epoch", fontsize=24)#
##    plt.legend(loc="upper right")
##    plt.ylim(0.00, 0.025)

####
#### n_counts = np.negative(counts)
####           
#### counts_dSQ, bins_dSQ = np.histogram(dtest, bins = 17)  # counts_dSQ = randomly generated influent flow rate
#### counts_dSQ_r =np.asarray([x/20 for x in counts_dSQ]) # Reduced flow rate, divided by number of replicates         
####
#### plt.figure(figsize=(16,6))
#### #label = ['Experiment', 'Simulation']
#### plt.hist(Qr.T,bins=17, density=False, histtype='stepfilled')  #Qr.T
#### plt.gca().invert_yaxis()
#### plt.hist(dSQ_test.T, bins=17, density=False, histtype='stepfilled') 
#### # plt.legend(label)
#### plt.xlabel('Influent flow rate [l/day]')
#### plt.ylabel('Bin count [days]')
#### # plt.title('Experimental and Simulation influent flow rate profiles', fontsize='large')
#### # plt.savefig('C:/Users/mark_/Documents/02_Professional/03_Unil/SATW/MDPI_paper/MDPI_article/MDPI_article_v12/figures/F5_q_exp_sim.svg', format="svg")
#### plt.show()
#### x1 = np.random.normal(1, 2, 5000)
#### y1 = np.random.normal(-1, 3, 2000)  
#### aQr = np.arange(170) #,[0,169,1] dtype="float64"
#### for i in len(aQr):
####     bQr = aQr[i]
####        
#### aQr =  np.asarray(Qr, order="C") 
####
#### dSQ_r_mean =np.asarray([x/20 for x in dSQ_t]) # Reduced flow rate, divided by number of replicates 
#### b = dSQ_r_mean.flatten()
#### c = dSQ_t.flatten()
#### a = dSQ_t.mean(axis=0).flatten()
#### figX, ax = plt.subplots()
#### plt.hist([aQr.flatten(), c], bins, label=['Experiment', 'Simulation'])  #dSQ_t.mean(axis=0).T.flatten()
#### # plt.hist([x1, y1], bins, label=['x', 'y'])
#### # dSQ_test_n = np.negative(dSQ_test)# This works, don't touch
#### # p1 = plt.hist(Qr.T,bins=17, density=False, histtype='bar', cumulative = False) # This works, don't touch
#### # p1 = plt.hist(bins[:-1], bins, weights=counts, alpha=0.5, align='left')  #Experimental influent flow rate
#### # p2 = plt.hist(dSQ_test_n.T, bins=17, density=False, histtype='bar', cumulative = False) # This works, don't touch
#### # p2 = plt.hist(bins[:-1], bins_dSQ, weights=counts_dSQ_r, alpha=0.5, align='left')  #Simulation influent flow rate
#### ax.set_ylabel('Bin count [days]')
#### ax.set_xlabel('Influent flow rate [l/day]')
#### plt.legend()
#### # plt.savefig('C:/Users/mark_/Documents/02_Professional/03_Unil/SATW/MDPI_paper/MDPI_article/MDPI_article_v15/figures/F8_inf_binned.svg', format="svg")
#### plt.show()
####

####   Box plot alternative to be developed
##
##import numpy as np
##np.random.seed(19680801)
##import matplotlib.pyplot as plt
##
##fig, ax = plt.subplots()
####for color in ['tab:blue', 'tab:orange', 'tab:green']:
####    n = 750
####    x, y = np.random.rand(2, n)
####    scale = 200.0 * np.random.rand(n)
####    ax.scatter(x, y, c=color, s=scale, label=color,
####               alpha=0.3, edgecolors='none')
##x = [1,2,3,4,5,6,7,8]
##y = [10, 20, 30, 40, 50, 60, 70, 80]
##SD = ['8','s','p']
##level=0
##for level in range(3):
##    #for level in range(3):
##    #scale = 200.0 * np.random.rand(n)  # must have same size as x and y
##    ax.scatter(x, y, marker=SD[level])#,  , label=mark,
##               #alpha=0.3, edgecolors='none')  #s=scale,
##
##
###ax.legend()
##ax.grid(True)
##
##plt.show()
#####################


##############   F8_simulations  Make a VERTICLE box plot of the sorted simulation results    ###############
####Box extends from lower to upper quartile boundries (1st and 4th quartiles). Mean line = green, whis = 'range' => whiskers extend to extreme values. 
Rep_sort=[]
nexp=125  #The number of simulated experiments
for n in range(nexp):
    #Rep_sort.append(np.sort(Rep[n], axis = 0))
    Rep_sort.append(np.sort(Rep_rs[n], axis = 0))
Rep_sort = np.flip(np.asarray(Rep_sort))

plt.figure(figsize=(16.2, 10)) 
# plt.scatter(Q1, Repdf) #t[i][0:p]
# plt.plot(Rep_sort, marker='*', linestyle='None') 
flierprops = dict(marker='', markerfacecolor='g', markersize=4,
                  linestyle='none', markeredgecolor='k')

medianprops = dict(linestyle='', linewidth=0)
plt.boxplot(Rep_sort, whis = 100 , showmeans=True,  meanline=True, flierprops  =flierprops, notch=False, autorange = True, medianprops = medianprops) # ,  medianprops = dict(linestyle='', linewidth=0)meanpointprops = dict(linestyle='--', linewidth=0.5, color='purple'),,

# flierprops2 = dict(marker='o', markerfacecolor='r', markersize=40,
#                   linestyle='none', markeredgecolor='r')
# plt.boxplot(Rep2, whis = 2, showmeans=True, flierprops  =flierprops2,  meanline=True, notch=False, autorange = False, medianprops = medianprops) # ,
# # labels = ["a", "b", "c"]
# plt.legend(labels)
plt.grid(visible=True, which='major', axis='y') # 'both''
# plt.tick_params(axis='x', which ='both')
# x_ax =[i for i in range(126)]
# ticks = np.arange(min(x_ax), max(x_ax)+1, 5)
# plt.xticks(ticks, labels =ticks)#), step=5))
plt.xticks([])
plt.xlabel('125 simulated experiments', fontsize=24)
plt.ylabel('Cumulative calorific value reduction \n (Min-Max scaled)', fontsize=24)
#plt.title('Simulation using the MLP model - Predicted calorific value reduction', fontsize='large') #medium, x-large
#plt.savefig('C:/Users/mark_/Documents/02_Professional/03_Unil/SATW/MDPI_paper/MDPI_article/MDPI_article_v24/figures/Fig_simulations_box.svg', format="svg")
##plt.savefig('C:/Users/mark_/Documents/02_Professional/03_Unil/SATW/MDPI_paper/Journal communication/Sustainability/Round two special issue/Figures_originals/Fig_simulations_box.svg', format="svg", bbox_inches="tight", dpi=300)
##plt.savefig('C:/Users/mark_/Documents/02_Professional/03_Unil/SATW/MDPI_paper/Journal communication/Sustainability/Round two special issue/Figures_originals/Fig_simulations_box.pdf', format="pdf", bbox_inches="tight", dpi=300)
plt.show()

# ######################################   END MDPI paper script    ######################
##################################################################################################################
####################################################################################################################
####################################################################################################################
####import sys
####print(sys.executable)
####print(sys.version)


# ### Figure for information, not in MDPI article
# plt.figure(figsize=(16,6)) 
# plt.plot(y_train_mm, label = 'True outputs (Training data)')#, 'y'
# plt.plot(Yhat, label ='Predictions (Training data)' )#, 'y'[:,0]
# plt.legend()
# plt.xlabel('Day') #Measurement sequence number
# plt.ylabel('Prediction (Calorific value reduction (Min-Max scaled))')
# plt.title('MLP model training ', fontsize='large') #MLP predicted and observed flow rate during testing
# plt.show(); 

# ### Figure for information, not in MDPI article
# y_test_mm = pd.read_csv('C:/Users/mark_/userdata/Output/V15/y_test_mm_v3.csv', sep=',', decimal='.', header=None, index_col=False)
# Yhat_v = pd.read_csv('C:/Users/mark_/userdata/Output/V15/Yhat_v_v3.csv', sep=',', decimal='.', header=None, index_col=False)
# print(Yhat_v.describe())
# Yhat_v_m =Yhat_v.mean(axis=0).mean()
# Yhat_v_s =Yhat_v.std(axis=1).mean()
# print(Yhat_v_s/Yhat_v_m)


# plt.figure(figsize=(7,6))   
# plt.scatter(y_test_mm.iloc[:, [0]], Yhat_v,  c="k", s=1)
# plt.plot(xl, yl, c="k")
# # plt.legend()#['True outputs (Test data)', 'Predictions (Test data)', 'Predictions (Test-2 data)'])
# plt.xlabel('True values') #Measurement sequence number
# plt.ylabel('Predictions') 
# plt.show(); 

# reg = LinearRegression().fit(y_test_mm.iloc[:, [0]], Yhat_v) #, fit_intercept=True
# print("Validation test R2: %.2f" % reg.score(y_test_mm.iloc[:, [0]], Yhat_v))

# y_test_mm = pd.read_csv('C:/Users/mark_/userdata/Output/V15/y_test_mm_v3_noE2.csv', sep=',', decimal='.', header=None, index_col=False)
# Yhat_v = pd.read_csv('C:/Users/mark_/userdata/Output/V15/Yhat_v_v3_noE2.csv', sep=',', decimal='.', header=None, index_col=False)
# print(Yhat_v.describe())
# Yhat_v_m =Yhat_v.mean(axis=0).mean()
# Yhat_v_s =Yhat_v.std(axis=1).mean()
# print(Yhat_v_s/Yhat_v_m)

# plt.figure(figsize=(7,6))   
# plt.scatter(y_test_mm.iloc[:, [0]], Yhat_v,  c="k", s=1)
# plt.plot(xl, yl, c="k")
# # plt.legend()#['True outputs (Test data)', 'Predictions (Test data)', 'Predictions (Test-2 data)'])
# plt.xlabel('True values') #Measurement sequence number
# plt.ylabel('Predictions') 
# plt.show(); 

# reg = LinearRegression().fit(y_test_mm.iloc[:, [0]], Yhat_v) #, fit_intercept=True
# print("Validation test R2: %.2f" % reg.score(y_test_mm.iloc[:, [0]], Yhat_v))

########################################
#######  Scatter plots of results
# # ###  Check satw_data NN_preprocess line 2207 to see if shuffle is set to True or to False
 # ### Plot a slope of 1 to show a perfect correlation between predictions and true values
##yl = np.array([0,1]) #,0.1
##xl = np.array([0,1]) #,0.1
##
###### Shuffled and unshuffled data, Validation test
##y_test_mm = pd.read_csv('C:/Users/mark_/mark_data/Output/y_test_mm.csv', sep=',', decimal='.', header=None, index_col=False) #header=None,
##y_test_mm_ns = pd.read_csv('C:/Users/mark_/mark_data/Output/y_test_mm_ns.csv', sep=',', decimal='.', header=None, index_col=False)
##Yhat_polynomial_m = pd.read_csv('C:/Users/mark_/mark_data/Output/Yhat_polynomial_m.csv', sep=',', decimal='.', header=None, index_col=False) #header=None,
##Yhat_polynomial_m_ns = pd.read_csv('C:/Users/mark_/mark_data/Output/Yhat_polynomial_m_ns.csv', sep=',', decimal='.', header=None, index_col=False)
##Yhat_v_mlp = pd.read_csv('C:/Users/mark_/mark_data/Output/Yhat_v_mlp.csv', sep=',', decimal='.', header=None, index_col=False) #header=None,
##Yhat_v_mlp_ns = pd.read_csv('C:/Users/mark_/mark_data/Output/Yhat_v_mlp_ns.csv', sep=',', decimal='.', header=None, index_col=False)
##y_test_lstm = pd.read_csv('C:/Users/mark_/mark_data/Output/y_test_lstm.csv', sep=',', decimal='.', header=None, index_col=False) #header=None,
##Yhat_v_lstm = pd.read_csv('C:/Users/mark_/mark_data/Output/Yhat_v_lstm.csv', sep=',', decimal='.', header=None, index_col=False) #header=None,
##y_test_lstm_ns = pd.read_csv('C:/Users/mark_/mark_data/Output/y_test_lstm_ns.csv', sep=',', decimal='.', header=None, index_col=False) #header=None,
##Yhat_v_lstm_ns = pd.read_csv('C:/Users/mark_/mark_data/Output/Yhat_v_lstm_ns.csv', sep=',', decimal='.', header=None, index_col=False)
##
##S_res = pd.DataFrame(np.column_stack((y_test_mm, y_test_mm_ns, Yhat_polynomial_m, Yhat_polynomial_m_ns, Yhat_v_mlp, Yhat_v_mlp_ns)))#Yhat_v_lstm,
##S_res.columns = ['y_test_mm', 'y_test_mm_ns', 'Yhat_polynomial_m', 'Yhat_polynomial_m_ns', 'Yhat_v_mlp', 'Yhat_v_mlp_ns'] #'Yhat_v_lstm',, 'y_test_lstm_ns', 'Yhat_v_lstm_ns'
##
##L_res = pd.DataFrame(np.column_stack((y_test_lstm, Yhat_v_lstm, y_test_lstm_ns, Yhat_v_lstm_ns)))
##L_res.columns = ['y_test_lstm', 'Yhat_v_lstm', 'y_test_lstm_ns', 'Yhat_v_lstm_ns']
###print(S_res)
#######   Plot predicted versus true value  ########
##fig, ax = plt.subplots()
##mk = 5
##ax.plot(xl,yl,  c="k")
##ax.scatter(S_res['y_test_mm'], S_res['Yhat_polynomial_m'], s=mk, c='b', marker='v')# 4th degree polynomial
##ax.scatter(S_res['y_test_mm'], S_res['Yhat_v_mlp'], s=mk, c='r', marker='D') #  MLP
##ax.scatter(L_res['y_test_lstm'], L_res['Yhat_v_lstm'], s=mk, c='g', marker='s')  # LSTM
## # # ax.scatter(y_test_mm_t, Yhat_v, s=mk, c='k', marker='+')
##ax.set_title("Shuffled data")
##ax.set(xlabel='True values', ylabel='Predictions')
###fig.savefig('C:/Users/mark_/Documents/02_Professional/03_Unil/SATW/MDPI_paper/MDPI_article/MDPI_article_v24/figures/Fig_regression_shuffled.svg', format="svg")  
##plt.show()
##
## ##### UNshuffled data (ns = not shuffled), Validation test
## ###  Check satw_data NN_preprocess line 2207 to see if shuffle is set to True or to False 
## #####   Plot predicted versus true value  ########
##fig, ax = plt.subplots()
##mk = 5
##ax.plot(xl,yl,  c="k")
##ax.scatter(S_res['y_test_mm_ns'], S_res['Yhat_polynomial_m_ns'], s=mk, c='b', marker='v')# 4th degree polynomial
##ax.scatter(S_res['y_test_mm_ns'], S_res['Yhat_v_mlp_ns'], s=mk, c='r', marker='D') #  MLP
##ax.scatter(L_res['y_test_lstm_ns'], L_res['Yhat_v_lstm_ns'], s=mk, c='g', marker='s')  # LSTM
## # ax.scatter(y_test_mm_t, Yhat_v, s=mk, c='k', marker='+')
## # ax.set_title("Shuffled data")
##ax.set_title("Unshuffled data")
##ax.set(xlabel='True values', ylabel='Predictions')
###fig.savefig('C:/Users/mark_/Documents/02_Professional/03_Unil/SATW/MDPI_paper/MDPI_article/MDPI_article_v24/figures/Fig_regression_NOTshuffled.svg', format="svg")  
##plt.show()
##

###### LSTM Archives. The preprocessing for LSTM is done in the nn module.
####X_data_mm = []
####        # apply min-max scaling
####    for column in range(4): #df_norm.columns
####        X_data_mm.append([(X_data[:,column] - X_data[:,column].min()) / (X_data[:,column].max() - X_data[:,column].min())])
####    
####    X_data_mm = np.vstack(X_data_mm).T
####
####
####X_data_mm_rs = np.reshape(X_data_mm, (-1, t, X_data_mm.shape[1]))  # rs = reshaped for LSTM
####y_data_mm_rs = np.reshape(y_data_mm, (-1, t))#, y_data_mm.shape[1]))
#### Set shuffle equal to True or to False          
####inputs_train, outputs_train, inputs_test, outputs_test = train_test_split(X_data_mm_rs, y_data_mm_rs, test_size=0.2, random_state= 42, shuffle = True, stratify= None)  #_rs, shuffle = True, random_state=42, stratify= None
####scale_tanh = MinMaxScaler(feature_range=(-1,1))
####X_data_mm = scale_tanh.fit_transform(X_data)
####y_data_mm = scale_tanh.fit_transform(y_data)
####plt.figure(),plt.plot(X_data_mm), plt.show()
####plt.figure(),plt.plot(y_data_mm), plt.show()
####
####X_data_mm_tr, X_data_mm_ts, y_data_mm_tr, y_data_mm_ts = train_test_split(X_data_mm, y_data_mm, test_size=0.2, random_state= 42, shuffle = True, stratify= None)
####inputs_train, outputs_train, inputs_test, outputs_test = X_data_mm_tr, X_data_mm_ts, y_data_mm_tr, y_data_mm_ts 
####inputs_train, outputs_train, inputs_test, outputs_test = np.reshape(X_data_mm_tr, (-1, 1, X_data_mm_tr.shape[1])), np.reshape(y_data_mm_tr, (-1, 1)), np.reshape(X_data_mm_ts, (-1, 1, X_data_mm_ts.shape[1])), np.reshape(y_data_mm_ts, (-1, 1))
########inputs_train, outputs_train, inputs_test, outputs_test = afbr_lstm(X_data, y_data)  # What is this?
####inputs_train, outputs_train, inputs_test, outputs_test = X_train_mm, y_train_mm, X_test_mm, y_test_mm

######################################################################
######################################################################
########   Replicate Poly, MLP, lSTM Scatter plots and histograms

##poly_df = pd.read_csv(open("C:/Users/mark_/mark_data/Output/poly_corr_df.csv"), delimiter=',', decimal=',')
##mlp_df = pd.read_csv(open("C:/Users/mark_/mark_data/Output/mlp_corr_df.csv"), delimiter=',', decimal=',')
##lstm_df = pd.read_csv(open("C:/Users/mark_/mark_data/Output/lstm_corr_df.csv"), delimiter=',', decimal=',')
##poly_df = 
##mlp_df = np.genfromtxt(open("C:/Users/mark_/mark_data/Output/mlp_corr_df.csv"), delimiter=',')
##lstm_df = np.genfromtxt(open("C:/Users/mark_/mark_data/Output/lstm_corr_df.csv"), delimiter=',')

##x = np.concatenate([poly_df.loc[:,'Poly_rmse'].values,poly_df.loc[:,'Poly_rmse'].values, lstm_df.loc[:,'RMSE'].values])
##y = np.concatenate([poly_df.loc[:,'Poly_sl'].values, poly_df.loc[:,'Poly_sl'].values, lstm_df.loc[:,'Slope'].values])

##x = np.concatenate([poly_df.loc[:,'Poly_rmse'],poly_df.loc[:,'Poly_rmse'], lstm_df.loc[:,'RMSE']])
##y = np.concatenate([poly_df.loc[:,'Poly_sl'], poly_df.loc[:,'Poly_sl'], lstm_df.loc[:,'Slope']])
##
##x = np.concatenate([poly_df.loc[:,'Poly_rmse'],poly_df.loc[:,'Poly_rmse'], lstm_df.loc[:,'RMSE']])
##y = np.concatenate([poly_df.loc[:,'Poly_sl'], poly_df.loc[:,'Poly_sl'], lstm_df.loc[:,'Slope']])
##
##print('x', x)
##print('y',y)
 # # y = np.arange(0,len(x),1, dtype='float')
##plt.figure()
##plt.scatter(x,y)
##plt.show()
 # # plt.plot(x, marker='.', linestyle='')
 # # plt.hist(x, bins= 10, orientation='horizontal', stacked=True)

# Fixing random state for reproducibility
##np.random.seed(19680801)
##
#####some random data
##x = np.random.randn(1000)
##y = np.random.randn(1000)

### Use below to make the figure
##poly_s = np.genfromtxt(open("C:/Users/mark_/mark_data/Input/poly_sl.csv"), delimiter=',')  # poly_df.loc[:,'poly_s']
##poly_rmse = np.genfromtxt(open("C:/Users/mark_/mark_data/Input/poly_rmse.csv"), delimiter=',') #poly_df.loc[:,'poly_rmse']
##mlp_s = np.genfromtxt(open("C:/Users/mark_/mark_data/Input/mlp_sl.csv"), delimiter=',')  #mlp_df.loc[:,'mlp_s']
##mlp_rmse = np.genfromtxt(open("C:/Users/mark_/mark_data/Input/mlp_rmse.csv"), delimiter=',') #mlp_df.loc[:,'mlp_rmse'] 
##lstm_s = np.genfromtxt(open("C:/Users/mark_/mark_data/Input/lstm_sl.csv"), delimiter=',')  #lstm_df.loc[:,'lstm_s']
##lstm_rmse = np.genfromtxt(open("C:/Users/mark_/mark_data/Input/lstm_rmse.csv"), delimiter=',') #lstm_df.loc[:,'lstm_rmse']
##
##
##def scatter_hist(x, y, ax, ax_histx, ax_histy):  #
## #     # no labels
##    ax_histx.tick_params(axis="x", labelbottom=False)
##    ax_histy.tick_params(axis="y", labelleft=False)
##
## #     # the scatter plot:
##    #ax.scatter(x, y)
##    ax.scatter(poly_rmse, poly_s, s=4, c='b', marker='.')
##    ax.scatter(mlp_rmse, mlp_s, s=4, c='r', marker='.')
##    ax.scatter(lstm_rmse, lstm_s, s=4, c='g', marker='.')
##    ax.set_xlabel('RMSE')
##    ax.set_ylabel('Slope')
##
## #     # now determine nice limits by hand:
##    binwidth = 0.01
##    xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
##    lim = (int(xymax/binwidth) + 1) * binwidth
##
##    bins = np.arange(0, lim + binwidth, binwidth)   # replace -lim with 0
##    ax_histx.hist(poly_rmse, bins=bins, color='b')
##    ax_histy.hist(poly_s, bins=bins, orientation='horizontal', color='b')
##    ax_histx.hist(mlp_rmse, bins=bins, color='r')
##    ax_histy.hist(mlp_s, bins=bins, orientation='horizontal', color='r')
##    ax_histx.hist(lstm_rmse, bins=bins, color='g')
##    ax_histy.hist(lstm_s, bins=bins, orientation='horizontal', color='g')
##
## # # start with a square Figure
##fig = plt.figure(figsize=(8, 8))
##
## # # Add a gridspec with two rows and two columns and a ratio of 2 to 7 between
## # # the size of the marginal axes and the main axes in both directions.
## # # Also adjust the subplot parameters for a square plot.
##gs = fig.add_gridspec(2, 2,  width_ratios=(7, 2), height_ratios=(2, 7),
##                     left=0.1, right=0.9, bottom=0.1, top=0.9,
##                     wspace=0.05, hspace=0.05)
##
##ax = fig.add_subplot(gs[1, 0])
##ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
##ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)
##
## # # use the previously defined function
##scatter_hist(poly_rmse, poly_s, ax, ax_histx, ax_histy)
##scatter_hist(mlp_rmse, mlp_s, ax, ax_histx, ax_histy)
###scatter_hist(lstm_rmse, lstm_s, ax, ax_histx, ax_histy)
###fig.suptitle('Polynomial, MLP, and LSTM models. 100 replications. Slope vs RMSE')  #
####plt.savefig('C:/Users/mark_/Documents/02_Professional/03_Unil/SATW/MDPI_paper/complementary_information/figures/Fig_s_vs_rmse.svg', format='svg')
####plt.savefig('C:/Users/mark_/Documents/02_Professional/03_Unil/SATW/MDPI_paper/Journal communication/Sustainability/Round two special issue/Figures_originals/Fig_Qinf_binned.svg', format="svg", bbox_inches="tight", dpi=300)
##plt.show()

##
### a = x[0:100]
### ## 
### colors=["red" for i in range(nrep)]+["green" for i in range(nrep)]  #+["blue" for i in range(50)] 
##
### fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(12,12), sharex=False, sharey=False, tight_layout=True, gridspec_kw={'width_ratios': [5, 1], 'height_ratios': [1, 5]})
##
### axs[1,0].scatter(x,y, c=colors)
### axs[1,0].set_xlabel('RMSE')
### axs[1,0].set_ylabel('Slope of the regression line')
##
### n_bins=10
### axs[0,0].hist(x[0:nrep], bins=2 , orientation='vertical', stacked=False, color="red")
### axs[0,0].hist(x[nrep:200], bins= n_bins, orientation='vertical', stacked=False, color="green")  # for i in range(n_bins)]+["green" for i in range(n_bins)])
### axs[0,0].set_ylabel('Bin count')
##
### axs[1,1].hist(y[0:nrep], bins= n_bins, orientation='horizontal', stacked=False, color="red") # for i in range(n_bins)]+["green" for i in range(n_bins)])
### axs[1,1].hist(y[nrep:200], bins= n_bins, orientation='horizontal', stacked=False, color="green") 
### axs[1,1].set_xlabel('Bin count')
##
### axs[0,1].axis('off')
##
### # fig.savefig("test.png")
### # plt.savefig('C:/Users/mark_/Documents/02_Professional/03_Unil/SATW/MDPI_paper/MDPI_article/MDPI_article_v24/figures/Fig_reps_slope_rmse.svg', format="svg")
### plt.suptitle('Poly, MLP, LSTM, slope and RMSE')
### plt.show()
##    




# ############    MDPI figure 7 ###########
# #Make a Violin plot of the simulation results  #####
# # import seaborn as sns
# # fig, ax = plt.subplots()
# # ax.violinplot(Exp,Rep[1]) 
# # # ax.sns.violinplot(x=Exp,y=Rep2[1],color='g') 
# # plt.violinplot(Rep, Exp, points=75, widths=0.7, vert=False, showmeans=True,
# #                       showextrema=True, showmedians=False, bw_method=0.5)
# # # plt.violinplot(Rep2,Exp, points=75, widths=0.7, vert=False, showmeans=True,
# # #                       showextrema=True, showmedians=False, bw_method=0.5, color='r')
# # # # r['cmeans'].set_color('b')
# # # # plt.xticks(np.arange(1, 126, step=1)) #L9 ->28, L243 ->244
# # # plt.yticks(np.arange(0, 126, step=5)) #L9 ->28, L243 ->244
# # # plt.xlabel('Cumulative (Min-Max scaled) daily calorific value reduction', fontsize=12)
# # # plt.ylabel('Experiment number', fontsize=12)
# # # plt.title('Simulation using the MLP model - Predicted daily calorific value reduction', fontsize=12) #'large', x-large
# # # plt.savefig('C:/Users/mark_/Documents/02_Professional/03_Unil/SATW/MDPI_paper/MDPI_article/MDPI_article_v12/figures/F8_simulations_violin.svg', format="svg")
# # plt.show()

# # # ##### Make a HORIZONTAL box plot of the simulation results  
 
# # plt.figure(figsize=(6,10)) 
# # # flierprops = dict(marker='o', markerfacecolor='k', markersize=8,
# # #                   linestyle='none', markeredgecolor='k')
# # # medianprops = dict(linestyle='', linewidth=0)

# # df=[]
# # df = pd.DataFrame(np.mean(Rep, axis = 0), index=[i for i in range(125)])#[(4, 1, 0.5), (4, 1, 1.8), (4, 1, 4), (4, 2, 0.5), (4, 2, 1.8), (4, 2, 4), (4, 3, 0.5), (4, 3, 1.8), (4, 3, 4), (18, 1, 0.5), (18, 1, 1.8), (18, 1, 4), (18, 2, 0.5), (18, 2, 1.8), (18, 2, 4), (18, 3, 0.5), (18, 3, 1.8), (18, 3, 4), (36, 1, 0.5), (36, 1, 1.8), (36, 1, 4), (36, 2, 0.5), (36, 2, 1.8), (36, 2, 4), (36, 3, 0.5), (36, 3, 1.8), (36, 3, 4)][4,18,36],[1,2,3],[0.5,1.8,4]
# # df.T.boxplot(vert=False, whis = 4, showmeans=True,   meanline=True, notch=False, autorange = False)#, flierprops  =flierprops,, medianprops = medianpropswhis = 2, showmeans=True, flierprops  =flierprops,  meanline=True, notch=False, autorange = False, medianprops = medianprops) # , medianprops = dict(linestyle='', linewidth=0)meanpointprops = dict(linestyle='--', linewidth=0.5, color='purple'),,

# # # plt.subplots_adjust(left=0.25)
# # # plt.show()
# # plt.grid(b=False)#(b=True, which='major', axis='both')
# # x_ax =[i for i in range(126)]
# # ticks = np.arange(min(x_ax), max(x_ax)+1, 5)
# # plt.yticks(ticks, labels =ticks)#), step=5))
# # # plt.tick_params(axis='x', which ='both')
# # # plt.xticks(np.arange(-0.001, 1.1, step=0.501), ('0',"0.5", "1")) #(np.arange(1, 28, step=1)),step=0.5
# # plt.xlabel('Cumulative calorific value reduction (Min-Max scaled)')
# # plt.ylabel('Simulated experiment number')
# # # plt.title('Simulation using the MLP model - Predicted methane production', fontsize='large') #medium, x-large
# # # plt.savefig('C:/Users/mark_/Documents/02_Professional/03_Unil/SATW/MDPI_paper/MDPI_article/MDPI_article_v12/figures/F8_simulations_box_h.svg', format="svg")
# # plt.show() 

# # # c=0
# # # Rep = pd.DataFrame(Rep)
# #     # Rep.to_csv('C:/Users/mark_/Anaconda3/envs/Tensorflow_v1/Mark/bg_sum_mlp.csv', sep=';', index=False, decimal=',')
    
# # # X_train_mm, y_train_mm, X_test_mm, y_test_mm, SX_test_mm= satw_mlp_v4.AFBR_tensors(X_data, y_data, SX_test)

# # print(scipy.stats.ttest_ind(Rep[:,8], Rep[:,7], axis=0, equal_var=True))  #, nan_policy='propagate'
# # print(Rep[:,26])

