import os
from shutil import copyfile
import shutil
import argparse

def copy_pars(in_path, out_path):
    for i in range(1950,2019):
        for m in [1, 4, 7, 10]:
            out_path2 = out_path + str(i) + "_M" + str(m) 
            in_path2  = in_path + str(i) + "_M" +str(m)
            print(in_path2)
            if os.path.exists(in_path2 + "/best_pars.pkl"):
                if not os.path.exists(out_path2):
                    os.makedirs(out_path2)

                shutil.copy(in_path2 + "/best_pars.pkl", out_path2 +"/prev_pars.pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_path", help="input path")
    parser.add_argument("--out_path", help="output path")
    args = parser.parse_args()

    copy_pars(args.in_path, args.out_path)
