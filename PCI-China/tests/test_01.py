import pytest
import sys
from src.data import *
from src.figures import *

def test_create_plotly():
    fig = create_plotly_figure(input = "./figures/pci.csv")

def test_proc_data():
    proc_embedding(
        input_file = "Tests/Data/Input/embedding_example.txt",
        output_path = "Tests/Data/Output/"
    )

    proc_pd(
        input  = "Tests/data/Output/2000_2010.pkl",
        create = 0,
        seed   = 1,
        k_fold = 10,
        output = "Tests/Data/Output/database.db",
        embedding = "Tests/Data/Output/"
    )

    proc_pd(
        input  = "Tests/data/Output/2011.pkl",
        create = 0,
        seed   = 2,
        k_fold = 10,
        output = "Tests/Data/Output/database.db",
        embedding = "Tests/Data/Output/"
    )
