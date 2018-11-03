import itertools, pathlib, pickle, copy, random, os, glob , sys
from time import time

import pandas as pd
import numpy as np
import tensorflow as tf

import sklearn 
from sklearn.metrics import precision_recall_fscore_support

import keras 
from keras import backend as K
from keras.preprocessing.sequence import pad_sequences
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding, CuDNNLSTM, CuDNNGRU,  GlobalMaxPooling1D, GlobalAveragePooling1D

from src.hyper_parameters import *

def recall(true_value, predicted_value):
    true_positives = K.sum(K.round(K.clip(true_value * predicted_value, 0, 1)))
    all_positives = K.sum(K.round(K.clip(true_value, 0, 1)))
    recall = true_positives / (all_positives + K.epsilon())
    return recall

def precision(true_value, predicted_value):
    true_positives = K.sum(K.round(K.clip(true_value * predicted_value, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(predicted_value, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def F1(true_value, pred_value):
    return( 2.0 * recall(true_value, pred_value) * precision(true_value, pred_value) / ( recall(true_value, pred_value) + precision(true_value, pred_value) + K.epsilon() ) )

def gen_candidate(x, bandwidth=0.1, type='int', min_value = None, max_value = None):
    r = random.uniform(-bandwidth, bandwidth)
    new_x = x * (1+r)

    if type == 'int':
        if x * bandwidth < 1 :
            new_x = x + random.choice([-1,0,1])
        else:
            new_x = round(new_x) 
        


    if min_value != None:
        new_x = max(new_x, min_value)       

    if max_value != None:
        new_x = min(new_x, max_value)    

    return(new_x)   

def update_hyper_pars(hyper_pars, bandwidth= 0.1):
    v = copy.deepcopy(hyper_pars.varirate)
    v['meta_neurons']    =  gen_candidate( v['meta_neurons']   , bandwidth = bandwidth , type = 'int' , min_value = 1)
    v['meta_dropout']    =  gen_candidate( v['meta_dropout']   , bandwidth = bandwidth , type = '' , min_value = 0, max_value=0.99)
    v['meta_layer']      =  gen_candidate( v['meta_layer']     , bandwidth = bandwidth , type = 'int' , min_value = 1)
    v['lstm1_max_len']    =  gen_candidate( v['lstm1_max_len']   , bandwidth = bandwidth , type = 'int' , min_value = 1)
    v['lstm1_neurons']    =  gen_candidate( v['lstm1_neurons']   , bandwidth = bandwidth , type = 'int' , min_value = 1)
    v['lstm1_dropout']    =  gen_candidate( v['lstm1_dropout']   , bandwidth = bandwidth , type = '' , min_value = 0, max_value=0.99)
    v['lstm1_layer']      =  gen_candidate( v['lstm1_layer']     , bandwidth = bandwidth , type = 'int' , min_value = 1)
    # v['lstm2_max_len']    =  gen_candidate( v['lstm2_max_len']   , bandwidth = bandwidth , type = 'int' , min_value = 1)
    # v['lstm2_neurons']    =  gen_candidate( v['lstm2_neurons']   , bandwidth = bandwidth , type = 'int' , min_value = 1)
    # v['lstm2_dropout']    =  gen_candidate( v['lstm2_dropout']   , bandwidth = bandwidth , type = '' , min_value = 0, max_value=0.99)
    # v['lstm2_layer']      =  gen_candidate( v['lstm2_layer']     , bandwidth = bandwidth , type = 'int' , min_value = 1)
    v['fc_neurons']       =  gen_candidate( v['fc_neurons']      , bandwidth = bandwidth , type = 'int' , min_value = 1)
    v['fc_dropout']       =  gen_candidate( v['fc_dropout']      , bandwidth = bandwidth , type = '' , min_value = 0, max_value=0.99)
    v['fc_layer']         =  gen_candidate( v['fc_layer']        , bandwidth = bandwidth , type = 'int' , min_value = 1)
    v['max_words']        =  gen_candidate( v['max_words']       , bandwidth = bandwidth , type = 'int' , min_value = 1)
    v['lr']               =  gen_candidate( v['lr']              , bandwidth = bandwidth , type = '' , min_value = 0.000001)
    v['n_embedding']      =  gen_candidate( v['n_embedding']     , bandwidth = bandwidth , type = 'int' , min_value = 1, max_value = 300)
    v['decay']            =  gen_candidate( v['decay']           , bandwidth = bandwidth , type = '' , min_value = 0)
    v['w']                =  gen_candidate( v['w']               , bandwidth = bandwidth , type = '' , min_value = 0)
    print(v)

    f = copy.deepcopy(hyper_pars.fixed)
    f['mod_id'] = str(round((time())))
    return hyper_parameters(v, f) 


def calc_f1_df(x): 
    j1,j2,F1,j3 =  precision_recall_fscore_support(x.Y,x.Y_hat)
    f1 = F1[1]
    return pd.Series([f1], index=['f1'])



def calc_prev_month(year, month, period=1):
    if period == 1 :
        if month == 1 :
            return year-1 , 12
        else :
            return year, month - 1 
    else :
        y,m = calc_prev_month(year, month, 1)
        return calc_prev_month(y, m, period - 1 )

def calc_next_month(year, month, period=1):
    if period == 1 :
        if month == 12 :
            return year+1 , 1
        else :
            return year, month + 1
    else :
        y,m = calc_prev_month(year, month, 1)
        return calc_prev_month(y, m, period + 1 )


def gen_filename(year, month):
    return str(year) + "_Q" + str(month)


def build_output_folder_structure(year_target, month_target, models_path, create=True):
    if not os.path.exists(models_path):
        os.makedirs(models_path)

    output_folder = models_path + str(year_target) + "_M" + str(month_target) + '/'
    history_folder = output_folder + '/history/'

    if create:
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        if not os.path.exists(history_folder):
            os.makedirs(history_folder)

    return history_folder , output_folder



