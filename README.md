Website: www.policychangeindex.com

Authors: [Julian TszKin Chan](http://sites.google.com/site/ctszkin/) and [Weifeng Zhong](http://www.weifengzhong.com)

Please email all comments/questions to ctszkin [AT] gmail.com or weifeng [AT] weifengzhong.com

What is the Policy Change Index (PCI) of China?
-----------------------------------------------
China's industrialization process has long been a product of government direction, be it coercive central planning or ambitious industrial policy. For the first time in the literature, we develop a quantitative indicator of China's policy priorities over a long period of time, which we call the Policy Change Index (PCI) of China. The PCI is a leading indicator that runs from 1951 to the most recent quarter and can be updated in the future. In other words, the PCI not only helps us understand the past of China's industrialization but also allows us to make short-term predictions about its future directions.

The design of the PCI has two building blocks: (1) it takes as input data the full text of the *People's Daily* --- the official newspaper of the Communist Party of China --- since it was founded in 1946; (2) it employs a set of machine learning techniques to "read" the articles and detect changes in the way the newspaper prioritizes policy issues.

The source of the PCI's predictive power rests on the fact that the *People's Daily* is at the nerve center of the China's propaganda system and that propaganda changes often precede policy changes. Before the great transformation from the central planning under Mao to the economic reform program after Mao, for example, considerable efforts were made by the Chinese government to promote the idea of reform, move public opinion, and mobilize resources toward the new agenda. Therefore, by detecting (real-time) changes in propaganda, the PCI is, effectively, predicting (future) changes in policy.

For details about the methodology and findings of this project, please see the following research paper:

- Chan and Zhong. 2018. "Reading China: Predicting Policy Change with Machine Learning." AEI Economics Working Paper [No. 2018-11](http://www.aei.org/wp-content/uploads/2018/10/Reading-China-AEI-WP.pdf) (latest version available [here](../blob/master/docs/Reading_China.pdf)).


Disclaimer
----------
Results will change as the underlying models improve. A fundamental reason for adopting open source methods in this project is so that people from all backgrounds can contribute to the models that our society uses to assess and predict changes in public policy; when community-contributed improvements are incorporated, the model will produce better results.


Getting Started
---------------
TODO


Data
----
The raw data of the *People's Daily*, which are not provided in this repository, should be placed in the sub-folder ```PCI-China/data/raw/pd/```. Each file in this sub-folder should contain one year-quarter of data, be named by the respective year-quarter, and be in the ```.pkl``` format. For example, the raw data for the first quarter of 2018 should be in the file ```2018_Q1.pkl```. Below is the list of column names and types of each raw data file:
```{python}
>>> df1 = pd.read_pickle("./PCI-China/data/raw/pd/2018_Q1.pkl")
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
where ```title``` and ```body``` are the Chinese texts of the title and body of each article.

The processed data of the *People's Daily*, which are not provided in this repository, should be placed in the sub-folder ```PCI-China/data/proc/by_year/```. Each file in this sub-folder should contain one year-quarter of data, be named by the respective year-quarter, and be in the ```.pkl``` format. For example, the processed data for the first quarter of 2018 should be in the file ```2018_Q1.pkl```. Provided with the raw data, the processed data are the output of running ```proc_data.sh```. Below is the list of column names and types of each processed data file:
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
where ```title_int``` and ```body_int``` are the word embeddings (numeric vectors) of the title and body of each article.


Citing the Policy Change Index (PCI)
------------------------------------

Please cite the source of your analysis as "Policy Change Index of China release #.#.#, author's calculations." If you wish to link to the PCI, http://www.policychangeindex.com is preferred. Additionally, we strongly recommend that you describe the input data used in your analysis and provide a link to the materials required to replicate it.
