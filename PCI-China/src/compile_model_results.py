import pickle, glob, os, sys, pathlib, copy, argparse
import pandas as pd
from src.hyper_parameters import *

def compile_model_results(model, root="./"):
    if model != "window_5_years_quarterly" and model != "window_10_years_quarterly" and model != "window_2_years_quarterly":
        print('Error: model must be "window_5_years_quarterly" or "window_10_years_quarterly" or "window_2_years_quarterly"' )
        sys.exit(1)

    listing = glob.glob(root + '/models/' + model + '/*/best_pars.pkl')

    dic_list = []
    for file in listing:
        tmp = hyper_parameters.load(file)
        dic_list.append(tmp.to_dictionary())

    df = pd.DataFrame(dic_list)
    df['diff'] = df.test_F1 - df.forecast_F1
    df['pci'] = abs(df.test_F1 - df.forecast_F1)

    if not os.path.exists(root + '/visualization/' +  model ):
        os.makedirs(root + '/visualization/' +  model )

    df.to_csv(root + '/visualization/' +  model + '/results.csv', index=False)

    # df.sort_values(by=["year_target","mt_target"], axis=1, ascending=True, inplace=True)

    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Model name: window_5_years_quarterly, window_10_years_quarterly, window_2_years_quarterly")
    parser.add_argument("--root", help="Root folder", default="./")

    args = parser.parse_args()

    if args.model != "window_5_years_quarterly" and args.model != "window_10_years_quarterly" and args.model != "window_2_years_quarterly":
        print('Error: model must be "window_5_years_quarterly" or "window_10_years_quarterly" or "window_2_years_quarterly"' )
        sys.exit(1)

    compile_model_results(args.model, args.root)
