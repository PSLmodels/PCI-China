import pandas as pd
from src.data import *
from src.utils import *

embedding = create_embedding_data_for_testing(
    input = "Data/Input/sgns_renmin_Word+Character+Ngram/sgns_renmin_Word+Character+Ngram.txt",
    nrows = 500,
    ncols = 100
)
embedding.to_csv("Tests/Data/Input/embedding_example.txt", index=False, sep = " ")

