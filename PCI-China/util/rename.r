path = "../models/window_10_years_quarterly"

all_files = dir(path, pattern = "best_pars.pkl", recursive = TRUE, full.names = TRUE)

library(foreach)
library(stringr)
foreach (i = all_files) %do%{
    from = i
    to = str_replace(i,"best_pars.pkl","prev_pars.pkl")
    file.rename(from,to)
}

all_files = dir(path, recursive = FALSE, full.names = TRUE)

foreach (i = all_files) %do%{
    from = i
    to = str_replace(i,"Q1","M1")
    to = str_replace(to,"Q2","M4")
    to = str_replace(to,"Q3","M7")
    to = str_replace(to,"Q4","M10")
    file.rename(from,to)
}
