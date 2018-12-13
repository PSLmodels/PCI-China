import os
from shutil import copyfile
import argparse

def copy_pars(in_path, out_path)
for i in range(1950,2019):
    for q in range(1,5):
        m = (q - 1 ) * 3 + 1 
        out_path2 = out_path + str(i) + "_M" + str(m) 
        in_path2  = in_path + str(i) + "_Q" +str(q)

        if os.path.exists(in_path2 + "/best_pars.pkl"):
            if not os.path.exists(out_path2):
                os.makedirs(out_path2)

            shutil.copy(in_path2 + "/best_pars.pkl", out_path2 +"/prev_pars.pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_path", help="input path")
    parser.add_argument("--out_path", help="output path")
