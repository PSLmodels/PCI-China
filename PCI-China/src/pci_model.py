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
from src.specs import *
from src.functions import *
from src.hyper_parameters import *

class pci_model:
    def __init__(self, hyper_pars, embedding_matrix):
        self.hyper_pars = hyper_pars
        self.embedding_matrix = embedding_matrix
        self.setup()

    def setup(self):
        df = pci_model.read_data(
            data_directory = self.hyper_pars.fixed['data_directory'], 
            year = self.hyper_pars.fixed['year_target'],
            month = self.hyper_pars.fixed['month_target'],
            window_month = self.hyper_pars.fixed['year_window'] * 12 
            )

        testing_df = df.loc[df['training_group'].isin(self.hyper_pars.fixed['testing_group'])]
        training_df = df.loc[df['training_group'].isin(self.hyper_pars.fixed['training_group'])]
        val_df = df.loc[df['training_group'].isin(self.hyper_pars.fixed['validation_group'])]

        self.Y_train , self.X_train, self.id_train = pci_model.prep_data(training_df, self.hyper_pars)
        self.Y_test , self.X_test, self.id_test  = pci_model.prep_data(testing_df, self.hyper_pars)
        self.Y_val , self.X_val, self.id_val  = pci_model.prep_data(val_df, self.hyper_pars)

        all_Y = np.concatenate( (self.Y_train ,self.Y_test , self.Y_val) , 0 )

        tmp_w = sklearn.utils.class_weight.compute_class_weight('balanced',np.unique(all_Y),np.squeeze(all_Y))
        self.W = dict()

        self.W[0] = tmp_w[0]
        if self.hyper_pars.varirate['w'] > 0:
            self.W[1] = tmp_w[1] * self.hyper_pars.varirate['w']
        else:
            self.W[1] = tmp_w[0]

        self.embedding_matrix = self.embedding_matrix[:,:(self.hyper_pars.varirate['n_embedding']+1)]


    @staticmethod
    def read_data(data_directory, year, month, window_month = 0):

        if window_month == 0 :
            filename = os.path.join(data_directory, gen_filename(year,month) + ".pkl")
            return pd.read_pickle(filename) 
        else:
            df = pd.DataFrame()

            for i in range(1, window_month+1):
                y,m = calc_prev_month(year, month, i)
                filename = os.path.join(data_directory, gen_filename(y,m) + ".pkl")
                if not os.path.exists(filename):
                    continue
                df = df.append( pd.read_pickle(filename), sort = True )

            return df

    @staticmethod
    def prep_data(df, hyper_pars):

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
        year2 = (df.month + 12*(df.year - 1946)) / 72
        meta = np.column_stack((weekday0,weekday1,year2, df.month, df.title_len*10/263 ,df.body_len*10/88879, df.n_articles_that_day*10/393 ,df.n_pages_that_day*10/127, df.n_frontpage_articles_that_day*10/27))

        all_text = df.title_int + df.body_int

        if hyper_pars.fixed['body_text_combined'] == 1 :
            X = [pad_sequences(df.title_int + df.body_int, maxlen=hyper_pars.varirate['lstm1_max_len'],padding='post', truncating='post'), meta]
        else:
            X = [pad_sequences(df.title_int, maxlen=hyper_pars.varirate['lstm1_max_len'],padding='post', truncating='post'), pad_sequences(df.body_int, maxlen=hyper_pars.varirate['lstm2_max_len'],padding='post', truncating='post'), meta]

        return Y, X, df.id

    @staticmethod
    def load(path,filename='', year_from='', year_to='' ):
        if filename == '':
            filename = str(year_from) + '_' + str(year_to) 

        with open(path + '/' + filename + '.pkl', 'rb') as f:
            x = pickle.load(f)

        mm = keras.models.load_model(path + '/' + filename + '.hd5', custom_objects={'precision': precision, 'recall' : recall, 'F1' : F1})
        x.model = mm 
        return x


    def save( self, filename='', path='./Output/'):
        mm = self.model 
        self.model = None
        folder = os.path.join(path)
        pathlib.Path(folder).mkdir(parents=True, exist_ok=True) 

        with open(folder + '/' + filename + '.pkl' , 'wb') as f:
            pickle.dump(self, f)
        mm.save(folder + '/' + filename + '.hd5')
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
            df = pci_model.read_data(
                data_directory = self.hyper_pars.fixed['data_directory'], 
                year = self.hyper_pars.fixed['year_target'],
                month = self.hyper_pars.fixed['month_target']
            )
            Y , X, id = pci_model.prep_data(df, self.hyper_pars)
            Y_hat = self.model.predict(X) 

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

    def export_prediction(self):
        df = pci_model.read_data(data_directory = self.hyper_pars.fixed['data_directory'], year = range(
                self.hyper_pars.fixed['year_target'] - self.hyper_pars.fixed['year_window'] + 1,  
                self.hyper_pars.fixed['year_target']
                ), month = self.hyper_pars.fixed['month_target']
            )

        testing_df = copy.deepcopy( df[df.training_group == self.hyper_pars.fixed['testing_group']] )
        training_df = copy.deepcopy( df[df.training_group != self.hyper_pars.fixed['testing_group']] )

        testing_df = testing_df[['id', 'year','month']]
        training_df = training_df[['id', 'year','month']]

        testing_df['Y'] = self.Y_test
        testing_df['Y_hat'] = (self.model.predict(self.X_test) > 0.5) + 0 

        training_df['Y'] = self.Y_train
        training_df['Y_hat'] = (self.model.predict(self.X_train) > 0.5) + 0 

        testing_df['type'] = 'testing'
        training_df['type'] = 'training'

        out = training_df.append(testing_df, sort = True)

        tmp = self.forecast_one_step_simple()
        tmp['type'] = 'one_step'
        out = out.append(tmp, sort = True)

        out['year_target'] =  self.hyper_pars.fixed['year_target']
        out['month_target'] =  self.hyper_pars.fixed['month_target']
        return(out)



def create_and_train_model(hyper_pars,gpu):
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu

    with open(hyper_pars.fixed['embedding_path'] , 'rb') as f:
        embedding_matrix = pickle.load(f)

    def model_fun(input):
        obj = input.hyper_pars
        input_title = Input(shape=(obj.varirate['lstm1_max_len'],))

        net_title = Embedding(input.embedding_matrix.shape[0] ,
                input.embedding_matrix.shape[1],
                weights=[input.embedding_matrix],
                input_length=obj.varirate['lstm1_max_len'],
                trainable=False)(input_title)

        for i in range(1, obj.varirate['lstm1_layer']):
            net_title = CuDNNGRU(obj.varirate['lstm1_neurons'], return_sequences=True)(net_title)
        net_title = CuDNNGRU(obj.varirate['lstm1_neurons'])(net_title)
        net_title = Dropout(obj.varirate['lstm1_dropout'])(net_title)

        input_meta = Input(shape=(  input.X_train[1].shape[1] ,))
        net_meta = Dense(obj.varirate['meta_neurons'], activation='relu')(input_meta)
        net_meta = Dropout(obj.varirate['meta_dropout'])(net_meta)

        for i in range(1,obj.varirate['meta_layer']):
            net_meta = Dense(obj.varirate['meta_neurons'], activation='relu')(net_meta)
            net_meta = Dropout(obj.varirate['meta_dropout'])(net_meta)


        net_combined = keras.layers.concatenate([net_title, net_meta])
        for i in range(1,obj.varirate['fc_layer']+1):
            net_combined = Dense(obj.varirate['fc_neurons'], activation='relu')(net_combined)
            net_combined = Dropout(obj.varirate['fc_dropout'])(net_combined)
        net_combined = Dense(1, activation='sigmoid')(net_combined)

        out = keras.models.Model(inputs=[input_title,input_meta], outputs=[net_combined] )

        return out


    my_model = pci_model(
        hyper_pars = hyper_pars,
        embedding_matrix = embedding_matrix)

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


def run_pci_model(year_target, month_target, i, gpu, model, root="../", T=0.01, discount=0.05, bandwidth = 0.2):
    print('################################################')
    print('year' + str(year_target) + '; month: ' + str(month_target))
    print('################################################')

    if model == "window_5_years":
        get_fixed = get_fixed_5_years
        gen_hyper_pars = gen_hyper_pars_5_years
    elif model == "window_10_years":
        get_fixed = get_fixed_10_years
        gen_hyper_pars = gen_hyper_pars_10_years
    else:
        print('Error: model must be "window_5_years" or "window_10_years"' )
        sys.exit(1)


    models_path = get_fixed(year_target, month_target, root)['model_folder']

    history_folder, curr_folder = build_output_folder_structure(year_target, month_target, models_path, create=True)
    gpu = str(gpu)

    ## if the best_pars, prev_pars, and model.hd5 are already in the folder:
    if not os.path.exists(curr_folder+'best_pars.pkl') :
        prev_y, prev_m = calc_prev_month(year_target, month_target)
        junk, prev_folder = build_output_folder_structure(prev_y, prev_m, models_path, create=False)

        if  os.path.exists(prev_folder+'best_pars.pkl') :
            best_hyper_pars = hyper_parameters.load(prev_folder + 'best_pars.pkl')
            best_hyper_pars.fixed = get_fixed(year_target, month_target, root)
        else:
            best_hyper_pars = gen_hyper_pars(year_target, month_target, root)

        my_model = create_and_train_model(best_hyper_pars, gpu)
        best_hyper_pars.perf  = my_model.summary()
        best_hyper_pars.save(curr_folder, 'best_pars.pkl')
        best_hyper_pars.save(curr_folder, 'prev_pars.pkl')
        best_hyper_pars.save(history_folder)
        
    if i == 1 :
        best_hyper_pars = hyper_parameters.load(curr_folder + 'best_pars.pkl')
        best_hyper_pars.fixed = get_fixed(year_target, month_target, root)

        prev_hyper_pars = best_hyper_pars
        prev_hyper_pars.save(curr_folder, 'prev_pars.pkl')

        new_hyper_pars = update_hyper_pars(prev_hyper_pars, bandwidth)
    else:
        best_hyper_pars = hyper_parameters.load(curr_folder + 'best_pars.pkl')
        best_hyper_pars.fixed = get_fixed(year_target, month_target, root)
        if (not os.path.exists(curr_folder+'prev_pars.pkl') ): 
            prev_hyper_pars = best_hyper_pars
        else:
            prev_hyper_pars = hyper_parameters.load(curr_folder + 'prev_pars.pkl')
            prev_hyper_pars.fixed = get_fixed(year_target, month_target, root)
        new_hyper_pars = update_hyper_pars(prev_hyper_pars, bandwidth)

    ## run model
    my_model = create_and_train_model(new_hyper_pars, gpu)
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

