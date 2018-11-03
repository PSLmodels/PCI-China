import os, sys, re, argparse, math, pickle
import numpy as np
import pandas as pd
import datetime as dt
from itertools import product
import jieba
from keras.preprocessing.text import text_to_word_sequence
import keras

def cut(x):
    return " ".join(jieba.cut(x))

def stratify_sample(n,k):
    num_seq = list(range(1,k+1)) * (math.ceil(n/k))
    return( np.random.choice(  num_seq[0:n], n, replace=False) )
def cal_len(x):
    return(len(re.sub('UNK','U',''.join(x) )))

def prepare_variables(data, embedding, year, month, k_fold, tokenizer ):
    data['title_seg'] = data.apply(lambda row: cut(row['title']), axis=1)
    data['body_seg'] = data.apply(lambda row: cut(row['body']), axis=1)
    del data["title"]
    del data["body"]

    data = data.dropna(subset=['page']).copy()
    assert data['page'].isnull().sum(axis = 0) == 0 

    data['year'] = data['date'].dt.year
    data['quarter'] = data['date'].dt.quarter
    data['month'] = data['date'].dt.month
    data['day'] = data['date'].dt.day
    data['weekday'] = data['date'].dt.dayofweek + 1

    ## Create useful variable for ML
    data['frontpage'] = np.where(data['page']==1, 1, 0)
    data['page1to3'] = np.where(data['page'].isin(range(1,4)), 1, 0)


    data['title_seg'] = data.title_seg.apply(lambda x : [ word if word in embedding.keys() else 'unk'  for word in text_to_word_sequence(x) ])
    data['body_seg'] = data.body_seg.apply(lambda x : [ word if word in embedding.keys() else 'unk'  for word in text_to_word_sequence(x) ])

    data['title_len'] = data["title_seg"].apply(cal_len)
    data['body_len'] = data["body_seg"].apply(cal_len)

    data['n_articles_that_day'] = data.groupby(['date'])['id'].transform('count')
    data['n_pages_that_day'] = data.groupby(['date'])['page'].transform(max)

    data['n_frontpage_articles_that_day'] = data.groupby(['date'])['frontpage'].transform(sum)

    ## Create Stratum  
    data = data.sort_values(by='id')
    np.random.seed(year * 100 + month)

    data['training_group'] = data.groupby(['frontpage','year','month'] )['frontpage'].transform(lambda x: stratify_sample(x.shape[0],k_fold) )

    data['title_int'] = tokenizer.texts_to_sequences(data.title_seg)
    data['body_int'] = tokenizer.texts_to_sequences(data.body_seg)
    del data["title_seg"]
    del data["body_seg"]

    return data

def proc_data(k_fold,path):
    if not os.path.exists(path+"/proc/"):
        os.makedirs(path+"/proc/")

    if not os.path.exists(path+"/proc/by_month/"):
        os.makedirs(path+"/proc/by_month/")

    ## Prepare embedding
    embedding_raw = pd.read_csv(path + "/raw/sgns_renmin_Word+Character+Ngram/sgns_renmin_Word+Character+Ngram.txt",
                            delim_whitespace = True,
                            header= None,
                            skiprows = 1,
                            quoting =3
                            )

    embedding = {}
    for index,i in embedding_raw.iterrows():
        word = i[0]
        coefs = i[1:]
        embedding[word] = coefs

    with open(path + '/proc/embedding.pkl' , 'wb') as f:
        pickle.dump(embedding, f)

    ## Prepare tokenizer
    all_text = [*embedding, 'unk']
    all_text = [i for i in all_text if type(i) is not float]
    tokenizer = keras.preprocessing.text.Tokenizer(filters='')
    tokenizer.fit_on_texts(all_text)
    with open(path + '/proc/tokenizer.pkl' , 'wb') as f:
        pickle.dump(tokenizer, f)

    ## Prepare embedding_matrix
    word_index = tokenizer.word_index
    embedding_matrix = np.zeros((len(word_index) + 1, 300))
    for word, i in word_index.items():
        embedding_vector = embedding.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    new_vec = np.zeros((embedding_matrix.shape[0],1))
    new_vec[word_index.get('unk')] = 1 

    embedding_matrix = np.concatenate((new_vec, embedding_matrix), axis=1)

    with open(path + '/proc/embedding_matrix.pkl' , 'wb') as f:
        pickle.dump(embedding_matrix, f)


    for y,m in product(range(1946, 2019), range(1,13)):
        print(str(y) + ' - ' + str(m))

        filename = str(y)+"_M"+str(m) +".pkl"

        if not os.path.exists(path + "/raw/pd/" + filename):
            continue

        data =  pd.read_pickle(path + "/raw/pd/" + filename)

        df = prepare_variables(data, embedding, y, m, k_fold, tokenizer )
        df.to_pickle(path + "/proc/by_month/" +str(y)+"_M"+str(m) +".pkl")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", help="Path to the data folder", default = "../data")
    parser.add_argument("--k_fold", help="Sample the data into k sub-samples. Define training, validation and testing data in the specification.", type=int, default = 5)
    args = parser.parse_args()


    proc_data(args.k_fold, args.data_path)
