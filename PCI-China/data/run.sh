python proc_embedding.py
python proc_pd.py --input="Input/pd/pd_1946_2017.pkl" --create=1 --seed=1 --k_fold=10 --output="Output/database.db"
python proc_pd.py --input="Input/pd/pd_2018_10.pkl" --create=0 --seed=2 --k_fold=10 --output="Output/database.db"
python proc_pd.py --input="Input/pd/pd_2018_Q1_to_Q3.pkl" --create=0 --seed=3 --k_fold=10 --output="Output/database.db"