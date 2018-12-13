import os, sys, re, argparse, math, pickle

import pandas as pd
import jieba
import sqlite3
import numpy as np
import datetime as dt
from nltk.tokenize import word_tokenize


def proc_gm(input, create, seed, k_fold, output):

    np.random.seed(seed)

    files = os.listdir(input)
    df = pd.DataFrame()
    for f in files:
        data = pd.read_csv(input + f, encoding="utf-8")
        df = df.append(data)
    df = df.drop_duplicates()

    ## Clean up the dates
    df["date"] = df["date"].apply(str)
    df.date[df.date == "19851´39"] = "19850629"
    df.date[df.date == "860428"] = "19860428"
    df.date[df.date == "10680109"] = "19680109"
    df.date[df.date == "18670301"] = "19670301"
    df.date[df.date == "39850603"] = "19850603"
    df.date[df.date == "19710001"] = "19710601"
    df.date[df.date == "19760019"] = "19760119"
    df.date[df.date == "19790031"] = "19790831"
    df.date[df.date == "19840800"] = "19840820"
    df.date[df.date == "19850500"] = "19850520"
    df.date[df.date == "19910800"] = "19910808"

    df["date"] = pd.to_datetime(pd.to_numeric(df["date"]), format = "%Y%m%d")

    ## Clean up the pages
    df["page"] = df["page"].apply(str)

    def extract_page_num(text):
        find = re.compile(r'^([^,;abcdp`]*).*')
        m = re.match(find, text)
        return m.group(1).strip()

    df['page'] = df['page'].apply(extract_page_num)

    df.page[df.page == "f6"] = "6"
    df.page[df.page == "l"] = "1"
    df.page[df.page == "I"] = "1"
    df.page[df.page == "s4"] = "4"
    df.page[df.page == "145"] = "1"
    df.page[df.page == "0"] = "n"

    df["page"] = pd.to_numeric(df["page"], errors = "coerce")

    # drop articles without page numbers
    df = df.dropna(subset=['page']).copy()
    assert df['page'].isnull().sum(axis = 0) == 0
    print(df.shape)


    ## Make up the article id's
    df = df.sort_values(by=['date',"page",'title'])
    df = df.reset_index()
    df["id"] = df.index + 1


    ## Generate strata
    df['frontpage'] = np.where(df['page']==1, 1, 0)
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month

    df = df.sort_values(by='id')

    def stratify_sample(n,k):
        num_seq = list(range(1,k+1)) * (math.ceil(n/k))
        return( np.random.choice(num_seq[0:n], n, replace=False) )

    df['strata'] = df.groupby(['frontpage','year','month'] )['frontpage'].transform(lambda x: stratify_sample(x.shape[0],k_fold) )


    ## Preprocess Spanish words
    def clean_up_token(text):
        text = re.sub("¨|­", "", str(text)) # remove special punctuations (upside-down ? and !)
        text = word_tokenize(text)
        text = [c for c in text if c.isalnum() or "".join(re.split(r"\.|,|-", c)).isalnum()]
        text = ["DIGITO" if "".join(re.split(r"\.|,|-", c)).isnumeric() else c for c in text]
        return " ".join(text)

    df['title_seg'] = df['title'].apply(clean_up_token)


    ## Pick out wanted columns
    df = df[['date', 'id', 'page', 'title', 'key_terms', 'strata', 'title_seg']]


    ## Export to SQL
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
    proc_gm(input=args.input, create=args.create, seed=args.seed, k_fold=args.k_fold, output=args.output)

