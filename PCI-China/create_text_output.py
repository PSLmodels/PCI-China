import pickle, glob, os, sys, pathlib, copy, argparse
import pandas as pd
from src.pci_model import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Model name")
    parser.add_argument("--year", help="year", default=1950, type=int)
    parser.add_argument("--month", help="month", default=1, type=int)
    parser.add_argument("--gpu", help="To year", default="0")

    args = parser.parse_args()
    
    create_text_output(model = args.model, year_month = str(args.year) + "_M" +  str(args.month) , gpu = args.gpu)

