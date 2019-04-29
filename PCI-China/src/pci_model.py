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
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding, CuDNNLSTM, CuDNNGRU,  GlobalMaxPooling1D, GlobalAveragePooling1D, GRU
from keras.preprocessing.text import text_to_word_sequence
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



def calc_prev_month(year, month, period ):
    if period == 1 :
        if month == 1 :
            return year-1 , 12
        else :
            return year, month - 1 
    else :
        y,q = calc_prev_month(year, month, 1)
        return calc_prev_month(y, q, period - 1 )



def build_output_folder_structure(year_target, mt_target, models_path, create=True):
    if not os.path.exists(models_path):
        os.makedirs(models_path)

    output_folder = models_path + str(year_target) + "_M" + str(mt_target) + '/'
    history_folder = output_folder + '/history/'

    if create:
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        if not os.path.exists(history_folder):
            os.makedirs(history_folder)

    return history_folder , output_folder


def compile_model_results(model, root="./"):

    listing = glob.glob(root + '/models/' + model + '/*/best_pars.pkl')

    dic_list = []
    for file in listing:
        tmp = hyper_parameters_load(file)
        dic_list.append(tmp.to_dictionary())

    df = pd.DataFrame(dic_list)
    df['diff'] = df.test_F1 - df.forecast_F1
    df['pci'] = abs(df.test_F1 - df.forecast_F1)

    if not os.path.exists(root + '/figures/' +  model ):
        os.makedirs(root + '/figures/' +  model )

    df.to_csv(root + '/figures/' +  model + '/results.csv', index=False)

    return df
    

class pci_model:
    def __init__(self, hyper_pars):
        self.hyper_pars = hyper_pars
        tokenizer = self.load_tokenizer()
        embedding = self.load_embedding()

        df = pci_model.read_data(
            data_text = self.hyper_pars.fixed['data_text'], 
            year = self.hyper_pars.fixed['year_target'],
            month = self.hyper_pars.fixed['mt_target'],
            month_apart = - self.hyper_pars.fixed['month_window'] 
            )

        testing_df = df.loc[df['strata'].isin(self.hyper_pars.fixed['testing_group'])]
        training_df = df.loc[df['strata'].isin(self.hyper_pars.fixed['training_group'])]
        val_df = df.loc[df['strata'].isin(self.hyper_pars.fixed['validation_group'])]

        forecast_df = pci_model.read_data(
            data_text = self.hyper_pars.fixed['data_text'], 
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
        self.X_train = [pad_sequences(self.rawX_train[0], maxlen=hyper_pars.varirate['lstm1_max_len'], padding='post', truncating='post'), self.rawX_train[1] ]
        self.X_test = [pad_sequences(self.rawX_test[0], maxlen=hyper_pars.varirate['lstm1_max_len'], padding='post', truncating='post'), self.rawX_test[1] ]
        self.X_val = [pad_sequences(self.rawX_val[0], maxlen=hyper_pars.varirate['lstm1_max_len'], padding='post', truncating='post'), self.rawX_val[1] ]
        self.X_forecast = [pad_sequences(self.rawX_forecast[0], maxlen=hyper_pars.varirate['lstm1_max_len'], padding='post', truncating='post'), self.rawX_forecast[1] ]



    def update_weight(self):
        self.W[0] = self.y_prop[0]
        if self.hyper_pars.varirate['w'] > 0:
            self.W[1] = self.y_prop[1] * self.hyper_pars.varirate['w']
        else:
            self.W[1] = self.y_prop[0]

    def load_embedding_matrix(self):
        with open(self.hyper_pars.fixed['embedding_matrix_path'] , 'rb') as f:
            embedding_matrix = pickle.load(f)
        return embedding_matrix[:,:(self.hyper_pars.varirate['n_embedding']+1)]

    def load_embedding(self):
        with open(self.hyper_pars.fixed['embedding_path'] , 'rb') as f:
            embedding = pickle.load(f)
        return embedding

    def load_tokenizer(self):
        with open(self.hyper_pars.fixed['tokenizer'] , 'rb') as f:
            tokenizer = pickle.load(f)
        return tokenizer






    @staticmethod
    def read_data(data_text, year, quarter=None, month=None,  month_apart = 0):
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

        conn = sqlite3.connect(data_text)
        df = pd.read_sql_query("select * from main where date >= Datetime('"+str(from_date)+" 00:00:00') and date < Datetime('"+str(to_date)+" 00:00:00')  ", conn)
        df['date'] = pd.to_datetime(df['date'], errors='coerce')

        conn.close()
        return df

    @staticmethod
    def prep_data(df, hyper_pars,embedding, tokenizer):
        df = copy.deepcopy(df)



        df['title_seg'] = df.title_seg.apply(text_to_word_sequence)
        df['body_seg'] = df.body_seg.apply(text_to_word_sequence)
        
        ## Create Stratum  
        df['title_int'] = tokenizer.texts_to_sequences(df.title_seg)
        df['body_int'] = tokenizer.texts_to_sequences(df.body_seg)

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
        meta = np.column_stack((
                    weekday0,
                    weekday1,
                    year2, 
                    df.month, 
                    df.title_len*10/263 ,
                    df.body_len*10/88879, 
                    df.n_articles_that_day*10/393 ,
                    df.n_pages_that_day*10/127, 
                    df.n_frontpage_articles_that_day*10/27))

        all_text = df.title_int + df.body_int

        if hyper_pars.fixed['body_text_combined'] == 1 :
            X = [df.title_int + df.body_int , meta]
        else:
            X = [df.title_int, meta]

        return Y, X, df.id

    @staticmethod
    def load(path):
        with open(path + 'data.pkl', 'rb') as f:
            x = pickle.load(f)

        mm = keras.models.load_model(path + 'model.hd5', custom_objects={'precision': precision, 'recall' : recall, 'F1' : F1})
        x.model = mm 

        pars = hyper_parameters_load(path + 'best_pars.pkl')

        x.set_hyper_pars(pars)

        return x


    def save( self, path='', filename='data.pkl'):
        mm = self.model 
        self.model = None
        with open(path + 'data.pkl' , 'wb') as f:
            pickle.dump(self, f)
        self.model = mm 

    def save_model(self, path):
        self.model.save(path + "/model.hd5")

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
        forecast_data = pd.DataFrame(data = self.id_forecast)
        forecast_data['Y'] = Y
        forecast_data['Y_hat'] = Y_hat
        forecast_data['Y_pred'] = Y_hat > 0.5


        testing_df = pci_model.read_data(
            data_text = self.hyper_pars.fixed['data_text'], 
            year = self.hyper_pars.fixed['year_target'],
            month = self.hyper_pars.fixed['mt_target'],
            month_apart = - self.hyper_pars.fixed['month_window'] 
            )

        forecast_df = pci_model.read_data(
            data_text = self.hyper_pars.fixed['data_text'], 
            year = self.hyper_pars.fixed['year_target'],
            month = self.hyper_pars.fixed['mt_target'],
            month_apart = self.hyper_pars.fixed['forecast_period'] 
            )

        testing_data = pd.merge(testing_data, testing_df, on='id', how='left')
        forecast_data = pd.merge(forecast_data, forecast_df, on='id', how='left')

        testing_data.drop(["title_seg","body_seg"], axis=1, inplace= True)
        forecast_data.drop(["title_seg","body_seg"], axis=1, inplace= True)

        return testing_data, forecast_data



def create_and_train_model(hyper_pars,gpu,path):
    if gpu != "-1":
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
            if gpu != "-1":
                net_title = CuDNNGRU(pars['lstm1_neurons'], return_sequences=True)(net_title)
            else:
                net_title = GRU(pars['lstm1_neurons'], return_sequences=True)(net_title)

        if gpu != "-1":
            net_title = CuDNNGRU(pars['lstm1_neurons'])(net_title)
        else:
            net_title = GRU(pars['lstm1_neurons'])(net_title)

        net_title = Dropout(pars['lstm1_dropout'])(net_title)

        input_meta = Input(shape=(  input_pci_model.X_train[1].shape[1] ,))
        net_meta = Dense(pars['meta_neurons'], activation='relu')(input_meta)
        net_meta = Dropout(pars['meta_dropout'])(net_meta)

        for i in range(1,pars['meta_layer']):
            net_meta = Dense(pars['meta_neurons'], activation='relu')(net_meta)
            net_meta = Dropout(pars['meta_dropout'])(net_meta)


        net_combined = keras.layers.concatenate([net_title, net_meta])
        for i in range(1,pars['fc_layer']+1):
            net_combined = Dense(pars['fc_neurons'], activation='relu')(net_combined)
            net_combined = Dropout(pars['fc_dropout'])(net_combined)
        net_combined = Dense(1, activation='sigmoid')(net_combined)

        out = keras.models.Model(inputs=[input_title,input_meta], outputs=[net_combined] )

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
    elif model == "window_2_years_quarterly":
        get_fixed = get_fixed_2_years_quarterly
        gen_hyper_pars = gen_hyper_pars_2_years_quarterly
    else:
        print('Error: model must be "window_5_years_quarterly" or "window_10_years_quarterly" or "window_2_years_quarterly"' )
        sys.exit(1)

    def save_best(pars, model, path):
        pars.save(path, 'best_pars.pkl')
        pars.save(path, 'prev_pars.pkl')
        model.save_model(path)
        # testing_data, forecast_data = model.summary_articles()
        # testing_data.to_excel(path + 'testing_data.xlsx')
        # forecast_data.to_excel(path + 'forecast_data.xlsx')



    models_path = get_fixed(year_target, mt_target, root)['model_folder']
    forecast_period = get_fixed(year_target, mt_target, root)['forecast_period']

    history_folder, curr_folder = build_output_folder_structure(year_target, mt_target, models_path, create=True)
    gpu = str(gpu)

    ## if the best_pars, prev_pars, and model.hd5 are already in the folder:
    if not (os.path.exists(curr_folder+'best_pars.pkl') & os.path.exists(curr_folder+'prev_pars.pkl')):
        prev_y, prev_m = calc_prev_month(year_target, mt_target,forecast_period)
        junk, prev_folder = build_output_folder_structure(prev_y, prev_m, models_path, create=False)

        if  os.path.exists(prev_folder+'best_pars.pkl') :
            best_hyper_pars = hyper_parameters_load(prev_folder + 'best_pars.pkl')
            best_hyper_pars.fixed = get_fixed(year_target, mt_target, root)
        else:
            best_hyper_pars = gen_hyper_pars(year_target, mt_target, root)

        my_model = create_and_train_model(best_hyper_pars, gpu, curr_folder)
        best_hyper_pars.perf  = my_model.summary()
        save_best(best_hyper_pars, my_model, curr_folder)
        # best_hyper_pars.save(curr_folder, 'best_pars.pkl')
        # best_hyper_pars.save(curr_folder, 'prev_pars.pkl')
        best_hyper_pars.save(history_folder)
        prev_hyper_pars = best_hyper_pars
    if  (not os.path.exists(curr_folder+'best_pars.pkl')) & (os.path.exists(curr_folder+'prev_pars.pkl')):
        best_hyper_pars = hyper_parameters_load(curr_folder + 'prev_pars.pkl')

        my_model = create_and_train_model(best_hyper_pars, gpu, curr_folder)
        best_hyper_pars.perf  = my_model.summary()
        # best_hyper_pars.save(curr_folder, 'best_pars.pkl')
        # best_hyper_pars.save(curr_folder, 'prev_pars.pkl')
        save_best(best_hyper_pars, my_model, curr_folder)
        best_hyper_pars.save(history_folder)
        prev_hyper_pars = best_hyper_pars
    if i == 1 :
        best_hyper_pars = hyper_parameters_load(curr_folder + 'best_pars.pkl')
        best_hyper_pars.fixed = get_fixed(year_target, mt_target, root)

        prev_hyper_pars = best_hyper_pars
        prev_hyper_pars.save(curr_folder, 'prev_pars.pkl')

        new_hyper_pars = update_hyper_pars(prev_hyper_pars, bandwidth)
    else:
        best_hyper_pars = hyper_parameters_load(curr_folder + 'best_pars.pkl')
        best_hyper_pars.fixed = get_fixed(year_target, mt_target, root)
        if (not os.path.exists(curr_folder+'prev_pars.pkl') ): 
            prev_hyper_pars = best_hyper_pars
        else:
            prev_hyper_pars = hyper_parameters_load(curr_folder + 'prev_pars.pkl')
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
        save_best(new_hyper_pars, my_model, curr_folder)



def create_text_output(model,year_month, gpu="0", root="./"):

    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)

    my_model = pci_model.load(root + "models/" + model + "/" + year_month + "/")
    testing_data, forecast_data = my_model.summary_articles()

    output_folder = root + "figures/" + model + "/" + 'articles_review'  + "/"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    testing_data.to_csv(output_folder +  year_month + '_testing_data.csv')
    forecast_data.to_csv(output_folder  +  year_month +'_forecast_data.csv')


def get_fixed_5_years_quarterly(year_target, mt_target, root = "./"):
    fixed = {
                'month_window' : 5 * 12, 
                'forecast_period' : 3, 
                'batch_size': 256,
                'patience' : 3,
                'epochs' : 100,
                'testing_group' : [1,2],
                'validation_group' : [3,4],
                'training_group' : [5,6,7,8,9,10],
                'data_text' : root + '/Data/Output/database.db', 
                'embedding_matrix_path' : root + '/Data/Output/embedding_matrix.pkl', 
                'embedding_path' : root + '/Data/Output/embedding.pkl', 
                'tokenizer' : root + "/Data/Output/tokenizer.pkl",
                'model_folder' : root + '/models/window_5_years_quarterly/',
                'year_target' : year_target,
                'mt_target' : mt_target,
                'body_text_combined' : 1,
                'frontpage' : 1, # 1:first page; 0: page1-3
                'mod_id' : str(round((time())))
            }    
    return fixed

def gen_hyper_pars_5_years_quarterly(year_target, mt_target, root):
    x = hyper_parameters(
        varirate ={
            'meta_layer' : 2,
            'meta_neurons' : 50,
            'meta_dropout' : 0.4,
            'lstm1_max_len' : 100,
            'lstm1_neurons' : 80 ,
            'lstm1_dropout' : 0.1 ,
            'lstm1_layer' : 5,
            'fc_neurons' : 40,
            'fc_dropout' : 0.3,
            'fc_layer' : 2,
            'max_words' : 10000,
            'lr' : 0.002,
            'n_embedding' : 150,
            'decay': 0.0001,
            'w': 0.3
        },
        fixed = get_fixed_5_years_quarterly(year_target, mt_target, root)
    )
    return x 

def get_fixed_10_years_quarterly(year_target, mt_target, root = "./"):
    fixed = {
                'month_window' : 10 * 12, 
                'forecast_period' : 3, 
                'batch_size': 256,
                'patience' : 3,
                'epochs' : 100,
                'testing_group' : [1,2],
                'validation_group' : [3,4],
                'training_group' : [5,6,7,8,9,10],
                'data_text' : root + '/Data/Output/database.db', 
                'embedding_matrix_path' : root + '/Data/Output/embedding_matrix.pkl', 
                'embedding_path' : root + '/Data/Output/embedding.pkl', 
                'tokenizer' : root + "Data/Output/tokenizer.pkl",
                'model_folder' : root + '/models/window_10_years_quarterly/',
                'year_target' : year_target,
                'mt_target' : mt_target,
                'body_text_combined' : 1,
                'frontpage' : 1, # 1:first page; 0: page1-3
                'mod_id' : str(round((time())))
            }    
    return fixed

def gen_hyper_pars_10_years_quarterly(year_target, mt_target, root):
    x = hyper_parameters(
        varirate ={
            'meta_layer' : 2,
            'meta_neurons' : 50,
            'meta_dropout' : 0.4,
            'lstm1_max_len' : 100,
            'lstm1_neurons' : 80 ,
            'lstm1_dropout' : 0.1 ,
            'lstm1_layer' : 5,
            'fc_neurons' : 40,
            'fc_dropout' : 0.3,
            'fc_layer' : 2,
            'max_words' : 10000,
            'lr' : 0.002,
            'n_embedding' : 150,
            'decay': 0.0001,
            'w': 0.3
        },
        fixed = get_fixed_10_years_quarterly(year_target, mt_target, root)
    )
    return x 

def get_fixed_2_years_quarterly(year_target, mt_target, root = "./"):
    fixed = {
                'month_window' : 2 * 12, 
                'forecast_period' : 3, 
                'batch_size': 256,
                'patience' : 3,
                'epochs' : 100,
                'testing_group' : [1,2],
                'validation_group' : [3,4],
                'training_group' : [5,6,7,8,9,10],
                'data_text' : root + '/Data/Output/database.db', 
                'embedding_matrix_path' : root + '/Data/Output/embedding_matrix.pkl', 
                'embedding_path' : root + '/Data/Output/embedding.pkl', 
                'tokenizer' : root + "Data/Output/tokenizer.pkl",
                'model_folder' : root + '/models/window_2_years_quarterly/',
                'year_target' : year_target,
                'mt_target' : mt_target,
                'body_text_combined' : 1,
                'frontpage' : 1, # 1:first page; 0: page1-3
                'mod_id' : str(round((time())))
            }    
    return fixed

def gen_hyper_pars_2_years_quarterly(year_target, mt_target, root):
    x = hyper_parameters(
        varirate ={
            'meta_layer' : 2,
            'meta_neurons' : 50,
            'meta_dropout' : 0.4,
            'lstm1_max_len' : 100,
            'lstm1_neurons' : 80 ,
            'lstm1_dropout' : 0.1 ,
            'lstm1_layer' : 5,
            'fc_neurons' : 40,
            'fc_dropout' : 0.3,
            'fc_layer' : 2,
            'max_words' : 10000,
            'lr' : 0.002,
            'n_embedding' : 150,
            'decay': 0.0001,
            'w': 0.3
        },
        fixed = get_fixed_2_years_quarterly(year_target, mt_target, root)
    )
    return x 
