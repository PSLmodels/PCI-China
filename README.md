Website: [policychangeindex.org](https://policychangeindex.org)

[![Build Status](https://travis-ci.com/PSLmodels/PCI.svg?branch=master)](https://travis-ci.com/PSLmodels/PCI) [![codecov](https://codecov.io/gh/PSLmodels/PCI/branch/master/graph/badge.svg)](https://codecov.io/gh/PSLmodels/PCI)

Authors: [Julian TszKin Chan](https://sites.google.com/site/ctszkin/) and [Weifeng Zhong](https://www.weifengzhong.com)

Please email all comments/questions to ctszkin [AT] gmail.com or weifeng [AT] weifengzhong.com

What is the Policy Change Index (PCI) for China?
-----------------------------------------------
China's industrialization process has long been a product of government direction, be it coercive central planning or ambitious industrial policy. For the first time in the literature, we develop a quantitative indicator of China's policy priorities over a long period of time, which we call the Policy Change Index (PCI) for China. The PCI is a leading indicator that runs from 1951 to the most recent quarter and can be updated in the future. In other words, the PCI not only helps us understand the past of China's industrialization but also allows us to make short-term predictions about its future directions.

The design of the PCI has two building blocks: (1) it takes as input data the full text of the *People's Daily* --- the official newspaper of the Communist Party of China --- since it was founded in 1946; (2) it employs a set of machine learning techniques to "read" the articles and detect changes in the way the newspaper prioritizes policy issues.

The source of the PCI's predictive power rests on the fact that the *People's Daily* is at the nerve center of the China's propaganda system and that propaganda changes often precede policy changes. Before the great transformation from the central planning under Mao to the economic reform program after Mao, for example, considerable efforts were made by the Chinese government to promote the idea of reform, move public opinion, and mobilize resources toward the new agenda. Therefore, by detecting (real-time) changes in propaganda, the PCI is, effectively, predicting (future) changes in policy.

For details about the methodology and findings of this project, please see the following research paper:

- Chan, Julian TszKin and Weifeng Zhong. 2019. "Reading China: Predicting Policy Change with Machine Learning." AEI Economics Working Paper [No. 2018-11](http://www.aei.org/wp-content/uploads/2018/10/Reading-China-AEI-WP.pdf) (latest version available [here](https://policychangeindex.org/Reading_China.pdf)).


Disclaimer
----------
Results will change as the underlying models improve. A fundamental reason for adopting open source methods in this project is so that people from all backgrounds can contribute to the models that our society uses to assess and predict changes in public policy; when community-contributed improvements are incorporated, the model will produce better results.


Usage
---------------
Three python functions and an R script are available for users to process data and build the neural network models to construct the PCI for China.

- `proc_pd.py`:              Process and prepare the raw data from the *People's Daily* for building the neural network models.
- `pci.py`:                    Train a neural network model to construct the PCI for a specified year-quarter.
- `compile_model_results.py`:  Compile the results from all models and export them to a `.csv` file.
- `generate_figures.r`:        Generate figures.

Users can check out the descriptions of the arguments for each python function using the `--help` option. For example:

```{shell}
./PCI-China/data>python proc_pd.py --help
Using TensorFlow backend.
usage: proc_pd.py [-h] [--input INPUT] [--create CREATE] [--seed SEED]
                  [--k_fold K_FOLD] [--output OUTPUT]

optional arguments:
  -h, --help       show this help message and exit
  --input INPUT    Input file
  --create CREATE  1:Create database 0:Append
  --seed SEED      Seed
  --k_fold K_FOLD  k_fold
  --output OUTPUT  Output filename
```

```{shell}
./PCI-China>python pci.py --help
Using TensorFlow backend.
usage: pci.py [-h] [--model MODEL] [--year YEAR] [--month MONTH] [--gpu GPU]
              [--iterator ITERATOR] [--root ROOT] [--temperature TEMPERATURE]
              [--discount DISCOUNT] [--bandwidth BANDWIDTH]

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL         Model name: window_5_years_quarterly,
                        window_10_years_quarterly, window_2_years_quarterly
  --year YEAR           Target year
  --month MONTH         Target month
  --gpu GPU             Which gpu to use
  --iterator ITERATOR   Iterator in simulated annealing
  --root ROOT           Root directory
  --temperature TEMPERATURE
                        Temperature in simulated annealing
  --discount DISCOUNT   Discount factor in simulated annealing
  --bandwidth BANDWIDTH
                        Bandwidth in simulated annealing```

```{shell}
./PCI-China>python compile_model_results.py --help
usage: compile_model_results.py [-h] [--model MODEL] [--root ROOT]

optional arguments:
  -h, --help     show this help message and exit
  --model MODEL  Model name: window_5_years_quarterly,
                 window_10_years_quarterly
  --root ROOT    Root folder
```

Data
----
The raw data of the *People's Daily*, which are not provided in this repository, should be placed in the sub-folder `PCI-China/Input/pd/`. Each file in this sub-folder should contain one year-quarter of data, be named by the respective year-quarter, and be in the `.pkl` format. For example, the raw data for the first quarter of 2018 should be in the file `2018_Q1.pkl`. Below is the list of column names and types of each raw data file:

```{python}
>>> df1 = pd.read_pickle("./PCI-China/Input/pd/pd_1946_1975.pkl")
>>> df1.dtypes
date     datetime64[ns]
year              int64
month             int64
day               int64
page              int64
title            object
body             object
id                int64
dtype: object
```

where `title` and `body` are the Chinese texts of the title and body of each article.

The processed data of the *People's Daily*, which are not provided in this repository, should be placed in the sub-folder `PCI-China/data/proc/by_year/`. Each file in this sub-folder should contain one year-quarter of data, be named by the respective year-quarter, and be in the `.pkl` format. For example, the processed data for the first quarter of 2018 should be in the file `2018_Q1.pkl`. Provided with the raw data, the processed data are the output of running `proc_data.sh`. Below is the list of column names and types of each processed data file:

```{python}
>>> df2 = pd.read_pickle("./PCI-China/data/proc/by_year/2018_Q1.pkl")
>>> df2.dtypes
date                             datetime64[ns]
year                                      int64
month                                     int64
day                                       int64
page                                      int64
id                                        int64
quarter                                   int64
weekday                                   int64
frontpage                                 int32
page1to3                                  int32
title_len                                 int64
body_len                                  int64
n_articles_that_day                       int64
n_pages_that_day                          int64
n_frontpage_articles_that_day             int32
training_group                            int32
title_int                                object
body_int                                 object
dtype: object
```

where `title_int` and `body_int` are the word embeddings (numeric vectors) of the title and body of each article.


Citing the Policy Change Index (PCI) for China
---------------------------------------------

Please cite the source of the latest Policy Change Index (PCI) for China by the website: https://policychangeindex.org.

For academic work, please cite the following research paper:

- Chan, Julian TszKin and Weifeng Zhong. 2019. "Reading China: Predicting Policy Change with Machine Learning." AEI Economics Working Paper [No. 2018-11](http://www.aei.org/wp-content/uploads/2018/10/Reading-China-AEI-WP.pdf) (latest version available [here](https://policychangeindex.org/Reading_China.pdf)).