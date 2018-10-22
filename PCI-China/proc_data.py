import argparse
from src import * 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", help="Path to the data folder", default = "./data")
    parser.add_argument("--k_fold", help="Sample the data into k sub-samples. Define training, validation and testing data in the specification.", type=int, default = 5)
    args = parser.parse_args()

    proc_data(args.k_fold, args.data_path)