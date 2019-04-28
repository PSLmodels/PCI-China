import pytest, sys, os
from src.data import *
from src.figures import *
from src.utils import *
from src.pci_model import *

def test_create_plotly():
    fig = create_plotly_figure(input = "./figures/pci.csv")

def test_proc_data():
    print(os.getcwd())

    if not os.path.exists("./tests/Data/Output/"):
        os.makedirs("./tests/Data/Output/")

    proc_embedding(
        input_file = "./tests/Data/Input/embedding_example.txt",
        output_path = "./tests/Data/Output/"
    )

    with open("./tests/Data/Output/embedding.pkl" , 'rb') as f:
        embedding = pickle.load(f)
        
    words = list( embedding.keys())

    example_data_2000 = gen_testing_data(words = words, from_year = 2000, to_year = 2010, type = 0, seed = 1 )
    example_data_2000.to_pickle("./tests/Data/Output/2000_2010.pkl")

    example_data_2011 = gen_testing_data(words = words, from_year = 2011, to_year = 2011, type = 1, seed = 2 )
    example_data_2011.to_pickle("./tests/Data/Output/2011.pkl")


    proc_pd(
        input  = "./tests/Data/Output/2000_2010.pkl",
        create = 0,
        seed   = 1,
        k_fold = 10,
        output = "./tests/Data/Output/database.db",
        embedding = "./tests/Data/Output/"
    )

    proc_pd(
        input  = "./tests/Data/Output/2011.pkl",
        create = 0,
        seed   = 2,
        k_fold = 10,
        output = "./tests/Data/Output/database.db",
        embedding = "./tests/Data/Output/"
    )


    run_pci_model(year_target=2011, mt_target=1, i=1, gpu=-1, model="window_2_years_quarterly", root = "./tests/", T=0.01, discount=0.05, bandwidth = 0.2 )
