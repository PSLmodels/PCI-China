import os, glob, sys, argparse, random
from time import time
import numpy as np
import tensorflow as tf
import keras
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding, CuDNNLSTM, CuDNNGRU,  GlobalMaxPooling1D, GlobalAveragePooling1D
from src.pci_model import *

if __name__ == "__main__":
    tf.set_random_seed(round(time()))
    np.random.seed(round(time()))
    random.seed(round(time()))

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Model name: window_5_years_quarterly, window_10_years_quarterly, window_2_years_quarterly")
    parser.add_argument("--year", help="Target year", type=int)
    parser.add_argument("--month", help="Target month", type=int)
    parser.add_argument("--gpu", help="Which gpu to use", default = 0)
    parser.add_argument("--iterator", help="Iterator in simulated annealing", type=int)
    parser.add_argument("--root", help="Root directory", default = "./")
    parser.add_argument("--temperature", help="Temperature in simulated annealing", default =0.01, type=float)
    parser.add_argument("--discount", help="Discount factor in simulated annealing", default =0.05, type=float)
    parser.add_argument("--bandwidth", help="Bandwidth in simulated annealing", default = 0.2, type=float)

    args = parser.parse_args()

    print("###############################")
    print("### Year: " + str(args.year) + " | Month: " + str(args.month) + " ###")
    print("###############################")
    print(args)
    print("###############################")

    if args.model != "window_5_years_quarterly" and args.model != "window_10_years_quarterly" and args.model != "window_2_years_quarterly":
        print('Error: model must be "window_5_years_quarterly" or "window_10_years_quarterly" or "window_2_years_quarterly"' )
        sys.exit(1)

    run_pci_model(args.year, args.month, args.iterator, args.gpu, model=args.model, root = args.root, T=args.temperature, discount=args.discount, bandwidth = args.bandwidth )

    