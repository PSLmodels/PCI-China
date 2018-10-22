import pickle, glob, os, sys, pathlib, copy, argparse
import pandas as pd
from src.hyper_parameters import *
from src.compile_model_results import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Model name: window_5_years, window_10_years")
    parser.add_argument("--root", help="Root folder", default="./")

    args = parser.parse_args()
    print("###")
    print(args.model)
    if args.model != "window_5_years" and args.model != "window_10_years":
        print('Error: model must be "window_5_years" or "window_10_years"' )
        sys.exit(1)

    compile_model_results(args.model)
