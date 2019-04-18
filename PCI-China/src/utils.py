import pandas as pd
import pickle
import numpy as np
import random 

def create_embedding_data_for_testing(input, nrows, ncols):
    embedding_raw = pd.read_csv(
        input,
        delim_whitespace = True,
        header= None,
        skiprows = 1,
        quoting =3,
        nrows = nrows
    )
    return embedding_raw.iloc[:,0:ncols]

def gen_df(dates, words, page):
    df = pd.DataFrame({'date' : dates})
    df['page'] = page
    df['title'] = gen_sentence(words, 5, len(df))
    df['body'] = gen_sentence(words, 10, len(df))
    return df

def gen_sentence(words, length, n ):
    out = list()
    for i in range(n):
        out.append(" ".join( random.sample(words,length)) )
    return out

def gen_testing_data(words, from_year, to_year, type, seed ):
    random.seed(seed)
    n = len(words)
    words0, words1 = words[:n//2] , words[n//2:]

    dates = pd.date_range(start= str(from_year) + "-01-01", end = str(to_year) + "-12-31")

    all_df = [
        gen_df(dates, words0, 1 + type), 
        gen_df(dates, words1, 2 - type), 
    ]

    out = pd.concat(all_df)
    out['id'] = np.arange(len(out)) + seed *  1000000
    return out 