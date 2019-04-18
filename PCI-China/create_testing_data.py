import pandas as pd
from src.data import *
from src.utils import *

embedding = create_embedding_data_for_testing(
    input = "Data/Input/sgns_renmin_Word+Character+Ngram/sgns_renmin_Word+Character+Ngram.txt",
    nrows = 500,
    ncols = 100
)
embedding.to_csv("Tests/Data/Input/embedding_example.txt", index=False, sep = " ")

proc_embedding(
    input_file = "Tests/Data/Input/embedding_example.txt",
    output_path = "Tests/Data/Output/"
)

with open("Tests/Data/Output/embedding.pkl" , 'rb') as f:
    embedding = pickle.load(f)
    
words = list( embedding.keys())

example_data_2000 = gen_testing_data(words = words, from_year = 2000, to_year = 2010, type = 0, seed = 1 )
example_data_2000.to_pickle("Tests/Data/Output/2000_2010.pkl")

example_data_2011 = gen_testing_data(words = words, from_year = 2011, to_year = 2011, type = 1, seed = 2 )
example_data_2011.to_pickle("Tests/Data/Output/2011.pkl")
