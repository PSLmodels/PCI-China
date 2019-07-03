from src.data import *

proc_embedding(
    input_file = "Data/Input/sgns_renmin_Word+Character+Ngram/sgns_renmin_Word+Character+Ngram.txt",
    output_path = "Data/Output/"
)

proc_pd(
    input  = "Data/Input/pd/pd_1946_1975.pkl",
    create = 1,
    seed   = 1,
    k_fold = 10,
    output = "Data/Output/database.db",
    embedding = "Data/Output/"
)

proc_pd(
    input  = "Data/Input/pd/pd_1976_2000.pkl",
    create = 0,
    seed   = 4,
    k_fold = 10,
    output = "Data/Output/database.db",
    embedding = "Data/Output/"
)

proc_pd(
    input  = "Data/Input/pd/pd_2001_2017.pkl",
    create = 0,
    seed   = 5,
    k_fold = 10,
    output = "Data/Output/database.db",
    embedding = "Data/Output/"
)

proc_pd(
    input  = "Data/Input/pd/pd_2018_Q1_to_Q3.pkl",
    create = 0,
    seed   = 3,
    k_fold = 10,
    output = "Data/Output/database.db",
    embedding = "Data/Output/"
)

proc_pd(
    input  = "Data/Input/pd/pd_2018_10.pkl",
    create = 0,
    seed   = 2,
    k_fold = 10,
    output = "Data/Output/database.db",
    embedding = "Data/Output/"
)

proc_pd(
    input  = "Data/Input/pd/pd_2018_11.pkl",
    create = 0,
    seed   = 201811,
    k_fold = 10,
    output = "Data/Output/database.db",
    embedding = "Data/Output/"
)



proc_pd(
    input  = "Data/Input/pd/pd_2018_12.pkl",
    create = 0,
    seed   = 201812,
    k_fold = 10,
    output = "Data/Output/database.db",
    embedding = "Data/Output/"
)

proc_pd(
    input  = "Data/Input/pd/2019_Q1.pkl",
    create = 0,
    seed   = 201901,
    k_fold = 10,
    output = "Data/Output/database.db",
    embedding = "Data/Output/"
)

proc_pd(
    input  = "Data/Input/pd/2019_Q2.pkl",
    create = 0,
    seed   = 201902,
    k_fold = 10,
    output = "Data/Output/database.db",
    embedding = "Data/Output/"
)
