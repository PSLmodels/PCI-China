import pytest
import sys
from src.data import *
from src.figures import *
import os 
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
