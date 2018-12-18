import itertools, pathlib, pickle, copy, random, os, glob , sys
from time import time
import datetime, monthdelta
import pandas as pd
import numpy as np
import tensorflow as tf
import sqlite3
import sklearn 
from sklearn.metrics import precision_recall_fscore_support

import keras 
from keras import backend as K
from keras.preprocessing.sequence import pad_sequences
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding, CuDNNLSTM, CuDNNGRU,  GlobalMaxPooling1D, GlobalAveragePooling1D
from src.specs import *
from src.functions import *
from src.hyper_parameters import *
from keras.preprocessing.text import text_to_word_sequence

class pci_model:
    def __init__(self, hyper_pars):
        self.hyper_pars = hyper_pars
        tokenizer = self.load_tokenizer()
        embedding = self.load_embedding()

        df = pci_model.read_data(
            data_gm = self.hyper_pars.fixed['data_gm'], 
            year = self.hyper_pars.fixed['year_target'],
            month = self.hyper_pars.fixed['mt_target'],
            month_apart = - self.hyper_pars.fixed['month_window'] 
            )

        testing_df = df.loc[df['strata'].isin(self.hyper_pars.fixed['testing_group'])]
        training_df = df.loc[df['strata'].isin(self.hyper_pars.fixed['training_group'])]
        val_df = df.loc[df['strata'].isin(self.hyper_pars.fixed['validation_group'])]

        forecast_df = pci_model.read_data(
            data_gm = self.hyper_pars.fixed['data_gm'], 
            year = self.hyper_pars.fixed['year_target'],
            month = self.hyper_pars.fixed['mt_target'],
            month_apart = self.hyper_pars.fixed['forecast_period'] 
            )

        self.Y_train , self.rawX_train, self.id_train = pci_model.prep_data(training_df, self.hyper_pars, embedding, tokenizer)
        self.Y_test , self.rawX_test, self.id_test  = pci_model.prep_data(testing_df, self.hyper_pars, embedding, tokenizer)
        self.Y_val , self.rawX_val, self.id_val  = pci_model.prep_data(val_df, self.hyper_pars, embedding, tokenizer)
        self.Y_forecast , self.rawX_forecast, self.id_forecast  = pci_model.prep_data(forecast_df, self.hyper_pars, embedding, tokenizer)

        all_Y = np.concatenate( (self.Y_train ,self.Y_test , self.Y_val) , 0 )

        self.y_prop  = sklearn.utils.class_weight.compute_class_weight('balanced',np.unique(all_Y),np.squeeze(all_Y))
        self.W = dict()
        self.set_hyper_pars(hyper_pars)


    def set_hyper_pars(self, hyper_pars):
        self.hyper_pars = hyper_pars
        self.update_weight() 
        self.X_train = [pad_sequences(self.rawX_train[0], maxlen=hyper_pars.varirate['lstm1_max_len'], padding='post', truncating='post'),  ]
        self.X_test = [pad_sequences(self.rawX_test[0], maxlen=hyper_pars.varirate['lstm1_max_len'], padding='post', truncating='post') ]
        self.X_val = [pad_sequences(self.rawX_val[0], maxlen=hyper_pars.varirate['lstm1_max_len'], padding='post', truncating='post')]
        self.X_forecast = [pad_sequences(self.rawX_forecast[0], maxlen=hyper_pars.varirate['lstm1_max_len'], padding='post', truncating='post')]



    def load_embedding_matrix(self):
        with open(self.hyper_pars.fixed['embedding_matrix_path'] , 'rb') as f:
            embedding_matrix = pickle.load(f)
        return embedding_matrix

    def load_embedding(self):
        with open(self.hyper_pars.fixed['embedding_path'] , 'rb') as f:
            embedding = pickle.load(f)
        return embedding

    def load_tokenizer(self):
        with open(self.hyper_pars.fixed['tokenizer'] , 'rb') as f:
            tokenizer = pickle.load(f)
        return tokenizer


    def update_weight(self):
        self.W[0] = self.y_prop[0]
        if self.hyper_pars.varirate['w'] > 0:
            self.W[1] = self.y_prop[1] * self.hyper_pars.varirate['w']
        else:
            self.W[1] = self.y_prop[0]




    @staticmethod
    def read_data(data_gm, year, quarter=None, month=None,  month_apart = 0):
        if ((quarter==None) == (month==None)):
            raise ValueError('Need specify either quarter or month.')

        if (quarter!=None):
            month = (quarter - 1 ) * 3 + 1 

        if month_apart < 0 :
            from_date = datetime.date(year, month, 1) + monthdelta.monthdelta(month_apart)
            to_date = datetime.date(year, month, 1)  
        elif month_apart > 0 :
            from_date = datetime.date(year, month, 1)  
            to_date  = datetime.date(year, month, 1) + monthdelta.monthdelta(month_apart)

        conn = sqlite3.connect(data_gm)
        df = pd.read_sql_query("select * from main where date >= Datetime('"+str(from_date)+" 00:00:00') and date < Datetime('"+str(to_date)+" 00:00:00')  ", conn)
        df['date'] = pd.to_datetime(df['date'], errors='coerce')

        conn.close()
        return df

    @staticmethod
    def prep_data(df, hyper_pars,embedding, tokenizer):
        df = copy.deepcopy(df)

        df['year'] = df['date'].dt.year
        df['quarter'] = df['date'].dt.quarter
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['weekday'] = df['date'].dt.dayofweek + 1

        ## Create useful variable for ML
        df['frontpage'] = np.where(df['page']==1, 1, 0)
        df['page1to3'] = np.where(df['page'].isin(range(1,4)), 1, 0)


        df['title_seg'] = df.title_seg.apply(lambda x : [ word if word in embedding.keys() else 'unk'  for word in text_to_word_sequence(x) ])
        ### body not available
        # df['body_seg'] = df.body_seg.apply(lambda x : [ word if word in embedding.keys() else 'unk'  for word in text_to_word_sequence(x) ])

        df['title_len'] = df["title"].str.len()
        ### body not available
        # df['body_len'] = df["body"].str.len()

        df['n_articles_that_day'] = df.groupby(['date'])['id'].transform('count')
        df['n_pages_that_day'] = df.groupby(['date'])['page'].transform(max)

        df['n_frontpage_articles_that_day'] = df.groupby(['date'])['frontpage'].transform(sum)

        ## Create Stratum  
        df['title_int'] = tokenizer.texts_to_sequences(df.title_seg)
        ### body not available
        # df['body_int'] = tokenizer.texts_to_sequences(df.body_seg)
        del df["title_seg"] #, df['body_seg']

        if hyper_pars.fixed['frontpage'] == 1 :
            Y = df.frontpage  
        else:
            Y = df.page1to3
        
        le = sklearn.preprocessing.LabelEncoder()
        Y = le.fit_transform(Y)
        Y = Y.reshape(-1,1)

        weekday1 = df.weekday <= 5 
        weekday0 = df.weekday > 5

        # TODO: Add backup for the normalization
        year2 = (df.month + 12*(df.year - 1965)) / 27
        # meta = np.column_stack(( df.year/df.year) )
                    # weekday0,
                    # weekday1,
                    # year2, 
                    # df.month))
                    # df.title_len*10/241 ,
                    ### body not available
                    # df.body_len*10/88879, 
                    # df.n_articles_that_day*10/42 ,
                    # df.n_pages_that_day*10/48, 
                    # df.n_frontpage_articles_that_day*10/15))

        all_text = df.title_int # + df.body_int

        ### body not available
        # if hyper_pars.fixed['body_text_combined'] == 1 :
        #     X = [pad_sequences(df.title_int + df.body_int, maxlen=hyper_pars.varirate['lstm1_max_len'], padding='post', truncating='post'), meta]
        # else:
        #     X = [pad_sequences(df.title_int, maxlen=hyper_pars.varirate['lstm1_max_len'], padding='post', truncating='post'), pad_sequences(df.body_int, maxlen=hyper_pars.varirate['lstm2_max_len'], padding='post', truncating='post'), meta]

        # X = [pad_sequences(df.title_int, maxlen=hyper_pars.varirate['lstm1_max_len'], padding='post', truncating='post'), meta]
        raw_X  = [df.title_int]

        return Y, raw_X, df.id

    # @staticmethod
    # def load(path,filename='', year_from='', year_to='' ):
    #     if filename == '':
    #         filename = str(year_from) + '_' + str(year_to) 

    #     with open(path + '/' + filename + '.pkl', 'rb') as f:
    #         x = pickle.load(f)

    #     mm = keras.models.load_model(path + '/' + filename + '.hd5', custom_objects={'precision': precision, 'recall' : recall, 'F1' : F1})
    #     x.model = mm 
    #     return x


    def save( self, path='', filename='data.pkl'):
        mm = self.model 
        self.model = None
        with open(path + 'data.pkl' , 'wb') as f:
            pickle.dump(self, f)

        self.model = mm 

    def summary_util(self, type):
        if type == "test":
            Y_hat = self.model.predict(self.X_test)
            Y = self.Y_test
        elif type == "train":
            Y_hat = self.model.predict(self.X_train) 
            Y = self.Y_train
        elif type == "val":
            Y_hat = self.model.predict(self.X_val)
            Y = self.Y_val
        elif type == "forecast":
            Y_hat = self.model.predict(self.X_forecast) 
            Y = self.Y_forecast

        Y_pred = Y_hat  > 0.5

        precision,recall,F1,junk = precision_recall_fscore_support(Y,Y_pred)
        out = dict()
        out['precision']=precision[1]
        out['recall']=recall[1]
        out['F1']=F1[1]

        return out


    def summary(self):
        out = dict()

        for i in ['test','train','val','forecast']:
            tmp = self.summary_util(i)
            for j in ['precision','recall','F1']:
                tmp[i+'_'+j] = tmp.pop(j)
            out.update(tmp)

        return out

    def summary_articles(self, root="./"):
        Y_hat_test = self.model.predict(self.X_test)
        testing_data = pd.DataFrame(data = self.id_test)
        testing_data['Y'] = self.Y_test
        testing_data['Y_hat'] = Y_hat_test
        testing_data['Y_pred'] = Y_hat_test > 0.5

        Y_hat = self.model.predict(self.X_forecast) 
        Y = self.Y_forecast
        forecast_data = pd.DataFrame(data = id)
        forecast_data['Y'] = Y
        forecast_data['Y_hat'] = Y_hat
        forecast_data['Y_pred'] = Y_hat > 0.5


        testing_df = pci_model.read_data(
            data_gm = self.hyper_pars.fixed['data_gm'], 
            year = self.hyper_pars.fixed['year_target'],
            month = self.hyper_pars.fixed['mt_target'],
            month_apart = - self.hyper_pars.fixed['month_window'] 
            )

        forecast_df = pci_model.read_data(
            data_gm = self.hyper_pars.fixed['data_gm'], 
            year = self.hyper_pars.fixed['year_target'],
            month = self.hyper_pars.fixed['mt_target'],
            month_apart = self.hyper_pars.fixed['forecast_period'] 
            )

        testing_data = pd.merge(testing_data, testing_df, on='id', how='left')
        forecast_data = pd.merge(forecast_data, forecast_df, on='id', how='left')


        return testing_data, forecast_data



def create_and_train_model(hyper_pars,gpu,path):
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu

    with open(hyper_pars.fixed['embedding_matrix_path'] , 'rb') as f:
        embedding_matrix = pickle.load(f)

    def model_fun(input_pci_model):
        pars = input_pci_model.hyper_pars.varirate
        input_title = Input(shape=(pars['lstm1_max_len'],))
        embedding_matrix = input_pci_model.load_embedding_matrix()

        net_title = Embedding(embedding_matrix.shape[0] ,
                embedding_matrix.shape[1],
                weights=[embedding_matrix],
                input_length=pars['lstm1_max_len'],
                trainable=False)(input_title)

        for i in range(1, pars['lstm1_layer']):
            net_title = CuDNNGRU(pars['lstm1_neurons'], return_sequences=True)(net_title)
        net_title = CuDNNGRU(pars['lstm1_neurons'])(net_title)
        net_title = Dropout(pars['lstm1_dropout'])(net_title)

        # input_meta = Input(shape=(  input_pci_model.X_train[1].shape[1] ,))
        # net_meta = Dense(pars['meta_neurons'], activation='relu')(input_meta)
        # net_meta = Dropout(pars['meta_dropout'])(net_meta)

        # for i in range(1,pars['meta_layer']):
        #     net_meta = Dense(pars['meta_neurons'], activation='relu')(net_meta)
        #     net_meta = Dropout(pars['meta_dropout'])(net_meta)


        # net_combined = keras.layers.concatenate([net_title, net_meta])
        # for i in range(1,pars['fc_layer']+1):
        #     net_combined = Dense(pars['fc_neurons'], activation='relu')(net_combined)
        #     net_combined = Dropout(pars['fc_dropout'])(net_combined)
        net_combined = Dense(1, activation='sigmoid')(net_title)

        out = keras.models.Model(inputs=[input_title], outputs=[net_combined] )

        return out

    ## If the data file exists, load it. Otherwise, create a new one and save.
    if os.path.exists(path + 'data.pkl') :
        with open(path + 'data.pkl' , 'rb') as f:
            my_model = pickle.load(f)
        my_model.set_hyper_pars(hyper_pars)
        my_model.model = model_fun(my_model)
    else :
        my_model = pci_model(hyper_pars = hyper_pars)
        my_model.set_hyper_pars(hyper_pars)
        my_model.model = model_fun(my_model)
        my_model.save(path=path, filename='data.pkl')    

    my_model.model = model_fun(my_model)
    my_model.model.compile(
        loss='binary_crossentropy', 
        optimizer=keras.optimizers.adam(my_model.hyper_pars.varirate['lr'], decay= my_model.hyper_pars.varirate['decay']), 
        metrics=[ precision, recall, F1])

    stop_cb = keras.callbacks.EarlyStopping(monitor='val_F1', patience=my_model.hyper_pars.fixed['patience'], mode='max')

    my_model.model.fit(
        my_model.X_train, 
        my_model.Y_train, 
        batch_size=my_model.hyper_pars.fixed['batch_size'], 
        epochs=my_model.hyper_pars.fixed['epochs'],
        validation_data=(my_model.X_val ,my_model.Y_val) , 
        shuffle=True, 
        class_weight=my_model.W,
        verbose = 1, 
        callbacks=[stop_cb])

    return my_model


def run_pci_model(year_target, mt_target, i, gpu, model, root="../", T=0.01, discount=0.05, bandwidth = 0.2):
    print('################################################')
    print('year' + str(year_target) + '; month: ' + str(mt_target))
    print('################################################')

    if model == "window_5_years_quarterly":
        get_fixed = get_fixed_5_years_quarterly
        gen_hyper_pars = gen_hyper_pars_5_years_quarterly
    elif model == "window_10_years_quarterly":
        get_fixed = get_fixed_10_years_quarterly
        gen_hyper_pars = gen_hyper_pars_10_years_quarterly
    else:
        print('Error: model must be "window_5_years_quarterly" or "window_10_years_quarterly"' )
        sys.exit(1)


    models_path = get_fixed(year_target, mt_target, root)['model_folder']

    history_folder, curr_folder = build_output_folder_structure(year_target, mt_target, models_path, create=True)
    gpu = str(gpu)

    ## if the best_pars, prev_pars, and model.hd5 are already in the folder:
    if not (os.path.exists(curr_folder+'best_pars.pkl') & os.path.exists(curr_folder+'prev_pars.pkl')):
        prev_y, prev_q = calc_prev_quarter(year_target, mt_target)
        junk, prev_folder = build_output_folder_structure(prev_y, prev_q, models_path, create=False)

        if  os.path.exists(prev_folder+'best_pars.pkl') :
            best_hyper_pars = hyper_parameters.load(prev_folder + 'best_pars.pkl')
            best_hyper_pars.fixed = get_fixed(year_target, mt_target, root)
        else:
            best_hyper_pars = gen_hyper_pars(year_target, mt_target, root)

        my_model = create_and_train_model(best_hyper_pars, gpu, curr_folder)
        best_hyper_pars.perf  = my_model.summary()
        best_hyper_pars.save(curr_folder, 'best_pars.pkl')
        best_hyper_pars.save(curr_folder, 'prev_pars.pkl')
        best_hyper_pars.save(history_folder)
        prev_hyper_pars = best_hyper_pars
    if  (not os.path.exists(curr_folder+'best_pars.pkl')) & (os.path.exists(curr_folder+'prev_pars.pkl')):
        best_hyper_pars = hyper_parameters.load(curr_folder + 'prev_pars.pkl')

        my_model = create_and_train_model(best_hyper_pars, gpu, curr_folder)
        best_hyper_pars.perf  = my_model.summary()
        best_hyper_pars.save(curr_folder, 'best_pars.pkl')
        best_hyper_pars.save(curr_folder, 'prev_pars.pkl')
        best_hyper_pars.save(history_folder)
        prev_hyper_pars = best_hyper_pars
    if i == 1 :
        best_hyper_pars = hyper_parameters.load(curr_folder + 'best_pars.pkl')
        best_hyper_pars.fixed = get_fixed(year_target, mt_target, root)

        prev_hyper_pars = best_hyper_pars
        prev_hyper_pars.save(curr_folder, 'prev_pars.pkl')

        new_hyper_pars = update_hyper_pars(prev_hyper_pars, bandwidth)
    else:
        best_hyper_pars = hyper_parameters.load(curr_folder + 'best_pars.pkl')
        best_hyper_pars.fixed = get_fixed(year_target, mt_target, root)
        if (not os.path.exists(curr_folder+'prev_pars.pkl') ): 
            prev_hyper_pars = best_hyper_pars
        else:
            prev_hyper_pars = hyper_parameters.load(curr_folder + 'prev_pars.pkl')
            prev_hyper_pars.fixed = get_fixed(year_target, mt_target, root)
        new_hyper_pars = update_hyper_pars(prev_hyper_pars, bandwidth)

    ## run model
    my_model = create_and_train_model(new_hyper_pars, gpu, curr_folder)
    new_hyper_pars.perf  = my_model.summary()
    new_hyper_pars.save(history_folder)

    print('################################################')
    print('iter: ' + str(i))
    print('F1 of new model: ' + str(new_hyper_pars.perf['val_F1'])  )
    print('F1 of previous model: ' + str(prev_hyper_pars.perf['val_F1'])  )
    print('F1 of the best model: ' + str(best_hyper_pars.perf['val_F1'])  )
    print('################################################')

    # if the new model out perform previous model 
    if ( new_hyper_pars.perf['val_F1'] + T * (1/(1+ i * discount)) >  prev_hyper_pars.perf['val_F1'] ):
        new_hyper_pars.save(curr_folder, 'prev_pars.pkl')

    # If it out perform the best model
    if ( (new_hyper_pars.perf['val_F1'] >  best_hyper_pars.perf['val_F1'])  ) :
        new_hyper_pars.save(curr_folder, 'best_pars.pkl')
        new_hyper_pars.save(curr_folder, 'prev_pars.pkl')


def create_text_output(year_target, mt_target, gpu, model, root="../"):
    if model == "window_5_years_quarterly":
        get_fixed = get_fixed_5_years_quarterly
        gen_hyper_pars = gen_hyper_pars_5_years
    elif model == "window_10_years_quarterly":
        get_fixed = get_fixed_10_years_quarterly
        gen_hyper_pars = gen_hyper_pars_10_years_quarterly
    elif model == "window_5_years_pp1to3":
        get_fixed = get_fixed_5_years_pp1to3
        gen_hyper_pars = gen_hyper_pars_10_years_pp1to3
    else:
        print('Error: model must be "window_5_years_quarterly", "window_10_years_quarterly", or "window_5_years_pp1to3"' )
        sys.exit(1)

    models_path = get_fixed(year_target, mt_target, root)['model_folder']

    history_folder, curr_folder = build_output_folder_structure(year_target, mt_target, models_path, create=True)
    gpu = str(gpu)

    best_hyper_pars = hyper_parameters.load(curr_folder + 'best_pars.pkl')
    my_model = create_and_train_model(best_hyper_pars, gpu, curr_folder)

    my_model.save("model", curr_folder)

    testing_data, forecast_data = my_model.summary_articles()

    testing_data.to_excel(curr_folder + 'testing_data.xlsx')
    forecast_data.to_excel(curr_folder + 'forecast_data.xlsx')
