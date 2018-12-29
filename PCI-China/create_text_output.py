import pickle, glob, os, sys, pathlib, copy, argparse
import pandas as pd
from src.pci_model import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Model name: window_5_years_quarterly, window_10_years_quarterly")
    parser.add_argument("--from_year", help="From year", default=1950, type=int)
    parser.add_argument("--to_year", help="To year", default=2018, type=int)
    parser.add_argument("--gpu", help="To year", default="0")

    args = parser.parse_args()
    print("###")
    print(args.model)
    if args.model != "window_5_years_quarterly" and args.model != "window_10_years_quarterly":
        print('Error: model must be "window_5_years_quarterly" or "window_10_years_quarterly"' )
        sys.exit(1)

    path = "./models/" + args.model + "/"


    for y in range(args.from_year, args.to_year+1):
        for q in [1, 4, 7, 10]:
            cur_path = path + str(y) + "_M" + str(q) + "/" 
            if os.path.exists(cur_path + "model.hd5"):
                create_text_output(cur_path, args.gpu)

                    



