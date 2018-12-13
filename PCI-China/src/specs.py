from time import time
from src.hyper_parameters import *

def get_fixed_5_years_quarterly(year_target, mt_target, root = "./"):
    fixed = {
                'month_window' : 6 * 12, 
                'forecast_period' : 3, 
                'batch_size': 256,
                'patience' : 3,
                'epochs' : 100,
                'testing_group' : [1,2],
                'validation_group' : [3,4],
                'training_group' : [5,6,7,8,9,10],
                'data_pd' : root + '/data/output/database.db', 
                'embedding_matrix_path' : root + '/data/output/embedding_matrix.pkl', 
                'embedding_path' : root + '/data/output/embedding.pkl', 
                'tokenizer' : root + "data/output/tokenizer.pkl",
                'model_folder' : root + '/models/window_5_years_quarterly/',
                'year_target' : year_target,
                'mt_target' : mt_target,
                'body_text_combined' : 1,
                'frontpage' : 1, # 1:first page; 0: page1-3
                'mod_id' : str(round((time())))
            }    
    return fixed

def gen_hyper_pars_5_years_quarterly(year_target, mt_target, root):
    x = hyper_parameters(
        varirate ={
            'meta_layer' : 2,
            'meta_neurons' : 50,
            'meta_dropout' : 0.4,
            'lstm1_max_len' : 100,
            'lstm1_neurons' : 80 ,
            'lstm1_dropout' : 0.1 ,
            'lstm1_layer' : 5,
            # 'lstm2_max_len' : 58,
            # 'lstm2_neurons' : 32,
            # 'lstm2_dropout' : 0.1,
            # 'lstm2_layer' : 2,
            'fc_neurons' : 40,
            'fc_dropout' : 0.3,
            'fc_layer' : 2,
            'max_words' : 10000,
            'lr' : 0.002,
            'n_embedding' : 150,
            'decay': 0.0001,
            'w': 0.3
        },
        fixed = get_fixed_5_years_quarterly(year_target, mt_target, root)
    )
    return x 


# def get_fixed_10_years(year_target, mt_target, root = "./"):
#     fixed = {
#                 'year_window' : 10, 
#                 'batch_size': 256,
#                 'patience' : 3,
#                 'epochs' : 100,
#                 'testing_group' : [1],
#                 'validation_group' : [2],
#                 'training_group' : [3,4,5],
#                 'data_directory' : root + '/data/proc/by_year', 
#                 'embedding_matrix_path' : root + '/data/proc/embedding_matrix.pkl', 
#                 'embedding_path' : root + '/data/proc/embedding_matrix.pkl', 
#                 'embedding_path' : root + '/data/proc/embedding_matrix.pkl', 
#                 'model_folder' : root + '/models/window_10_years/',
#                 'year_target' : year_target,
#                 'mt_target' : mt_target,
#                 'body_text_combined' : 1,
#                 'frontpage' : 1, # 1:first page; 0: page1-3
#                 'mod_id' : str(round((time())))
#             }    
#     return fixed

# def gen_hyper_pars_10_years(year_target, mt_target, root):
#     x = hyper_parameters(
#         varirate ={
#             'meta_layer' : 2,
#             'meta_neurons' : 50,
#             'meta_dropout' : 0.4,
#             'lstm1_max_len' : 100,
#             'lstm1_neurons' : 80 ,
#             'lstm1_dropout' : 0.1 ,
#             'lstm1_layer' : 5,
#             # 'lstm2_max_len' : 58,
#             # 'lstm2_neurons' : 32,
#             # 'lstm2_dropout' : 0.1,
#             # 'lstm2_layer' : 2,
#             'fc_neurons' : 40,
#             'fc_dropout' : 0.3,
#             'fc_layer' : 2,
#             'max_words' : 10000,
#             'lr' : 0.002,
#             'n_embedding' : 150,
#             'decay': 0.0001,
#             'w': 0.3
#         },
#         fixed = get_fixed_10_years(year_target, mt_target, root)
#     )
#     return x 
