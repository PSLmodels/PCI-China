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

    example_data_2000 = gen_testing_data(embedding = embedding, from_year = 2000, to_year = 2010, type = 0, seed = 1 )
    example_data_2000.to_pickle("./tests/Data/Output/2000_2010.pkl")

    example_data_2011 = gen_testing_data(embedding = embedding, from_year = 2011, to_year = 2012, type = 1, seed = 2 )
    example_data_2011.to_pickle("./tests/Data/Output/2011.pkl")


    proc_pd(
        input  = "./tests/Data/Output/2000_2010.pkl",
        create = 1,
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

    run_pci_model(year_target=2010, mt_target=3, i=1, gpu=-1, model="testing", root = "./tests/", T=0.01, discount=0.05, bandwidth = 0.2 )
    run_pci_model(year_target=2010, mt_target=4, i=1, gpu=-1, model="testing", root = "./tests/", T=0.01, discount=0.05, bandwidth = 0.2 )
    run_pci_model(year_target=2011, mt_target=1, i=2, gpu=-1, model="testing", root = "./tests/", T=0.01, discount=0.05, bandwidth = 0.2 )

    compile_model_results("testing", root = "./tests")

    create_text_output("testing", "2011_M1", gpu=-1, root ="./tests/")


# Verify PCI could identify the break
def test_verify_results():
    pci = pd.read_csv('./tests/figures/testing/results.csv')['pci']
    assert (pci[2] - pci[0] ) > 0.5
    assert (pci[2] - pci[1] ) > 0.5
