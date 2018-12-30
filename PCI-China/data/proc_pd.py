import os, sys, re, argparse, math, pickle

import pandas as pd
import jieba_fast as jieba
import sqlite3
import numpy as np
import datetime as dt
import math 
from keras.preprocessing.text import text_to_word_sequence

## Settings
# K_FOLD = 10
# SEED = 1 
# DATA_DIRECTORY = "./Input/pd"

def proc_pd(input, create, seed, k_fold, output ):

    df = pd.read_pickle(input)
    np.random.seed(seed)

    ## Remove advertisements
    print(df.shape)
    removed_text1 = ["广告", "广告更正", "广告　","　广告　","公益广告"] 
    df = df.loc[ ~ (( df['title'].isin(removed_text1)) |  ( df['body'].isin(removed_text1)) ) ] 
    print(df.shape)


    ## Remove duplicates
    print("Remove duplicates ")
    print(df.shape)
    df['dup'] = df.duplicated(subset=['title','body','date'], keep=False)
    df1 = df[~ ( df['dup'] & (df['body'].str.len() > 5))]
    df2 = df[(df['dup'] & (df['body'].str.len() > 5))]

    df2 = df2.sort_values(by=['date','body','title',"page"])
    df2 = df2[~df2.duplicated(subset=['title','body','date'], keep="first")]
    df = pd.concat([df1, df2])
    del df1, df2, df['dup']
    print(df.shape)

    ## Remove 第x版 until the first punctuation
    print("Remove 第x版 until the first punctuation")
    df['body'].str.replace('第.+?版.+?(　+| +|：|！|。|？)', '')

    ## Drop if null
    print("Drop if null")
    print(df.shape)
    print('##### Missing page number #####')
    print(df['page'].isnull().sum(axis = 0))
    df = df.dropna(subset=['page']).copy()

    assert df['page'].isnull().sum(axis = 0) == 0
    print(df.shape)

    ## Generate stratum
    print("Create stratum")
    df['frontpage'] = np.where(df['page']==1, 1, 0)
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month

    df = df.sort_values(by='id')

    def stratify_sample(n,k):
        num_seq = list(range(1,k+1)) * (math.ceil(n/k))
        return( np.random.choice(  num_seq[0:n], n, replace=False) )

    df['strata'] = df.groupby(['frontpage','year','month'] )['frontpage'].transform(lambda x: stratify_sample(x.shape[0],k_fold) )

    # del df['year'], df['month'], df['frontpage'] 

    ## Segmenting Chinese words/phrasers
    def cut(x):
        return " ".join(jieba.cut(x))

    print("jieba")
    df['title_seg'] = df.apply(lambda row: cut(row['title']), axis=1)
    df['body_seg'] = df.apply(lambda row: cut(row['body']), axis=1)

    print("Replacing unk for words that is not in the embedding")
    ## Replace word with unk if it is not in the embedding
    with open('./Output/embedding.pkl' , 'rb') as f:
        embedding = pickle.load(f)

    df['title_seg'] = df.title_seg.apply(lambda x : [ word if word in embedding.keys() else 'unk'  for word in text_to_word_sequence(x) ])
    df['body_seg'] = df.body_seg.apply(lambda x : [ word if word in embedding.keys() else 'unk'  for word in text_to_word_sequence(x) ])

    print("Create new variables")

    ## Create new variables
    df['quarter'] = df['date'].dt.quarter
    df['day'] = df['date'].dt.day
    df['weekday'] = df['date'].dt.dayofweek + 1

    df['page1to3'] = np.where(df['page'].isin(range(1,4)), 1, 0)

    df['title_len'] = df["title"].str.len()
    df['body_len'] = df["body"].str.len()

    df['n_articles_that_day'] = df.groupby(['date'])['id'].transform('count')
    df['n_pages_that_day'] = df.groupby(['date'])['page'].transform(max)

    df['n_frontpage_articles_that_day'] = df.groupby(['date'])['frontpage'].transform(sum)

    ## Keep variable
    df = df[['date', 'id', 'page', 'title', 'body', 'strata', 'title_seg','body_seg','year','quarter','month','day','weekday','frontpage','page1to3','title_len','body_len','n_articles_that_day','n_pages_that_day','n_frontpage_articles_that_day']]

    ## Export to SQL
    print("Export to SQL")
    idx1 = df.set_index(['date'])

    conn = sqlite3.connect(output)

    if (create == 1):
        idx1.to_sql("main", conn, if_exists="replace")
    elif (create == 0):
        idx1.to_sql("main", conn, if_exists="append")
    else:
        print("Didn't export output. Please specify create=1 or create=0")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help = "Input file", default = "")
    parser.add_argument("--create", help = "1:Create database 0:Append", type = int)
    parser.add_argument("--seed", help = "Seed" ,type = int)
    parser.add_argument("--k_fold", help = "k_fold" ,type = int)
    parser.add_argument("--output", help = "Output filename" )

    args = parser.parse_args()
    proc_pd(input=args.input, create=args.create, seed=args.seed, k_fold=args.k_fold, output=args.output)

