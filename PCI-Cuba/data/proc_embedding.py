import os, sys, re, argparse, math, pickle
import numpy as np
import pandas as pd
import datetime as dt
from itertools import product
import jieba
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.text import Tokenizer

data_directory = "Output/"
if not os.path.exists(data_directory):
    os.makedirs(data_directory)

def proc_embedding(input_file, output_path):
    print('Reading embedding file')
    embedding_raw = pd.read_csv(
        input_file,
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

    with open( output_path + '/embedding.pkl' , 'wb') as f:
        pickle.dump(embedding, f)

    print('Preparing tokenizer')
    ## Prepare tokenizer
    all_text = [*embedding, 'unk']
    all_text = [i for i in all_text if type(i) is not float]
    tokenizer = Tokenizer(filters='')
    tokenizer.fit_on_texts(all_text)
    with open(output_path +  '/tokenizer.pkl' , 'wb') as f:
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

    with open(output_path + '/embedding_matrix.pkl' , 'wb') as f:
        pickle.dump(embedding_matrix, f)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help = "Input file", default = "Input/SBW-vectors-300-min5/SBW-vectors-300-min5.txt")
    parser.add_argument("--output", help = "Output folder", default = "Output/")

    args = parser.parse_args()
    proc_embedding(args.input, args.output)
