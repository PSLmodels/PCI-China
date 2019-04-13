# # python create_text_output.py --model="window_5_years_quarterly" --year=2004 --month=1
# # python create_text_output.py --model="window_5_years_quarterly" --year=2013 --month=1
# python create_text_output.py --model="window_5_years_quarterly" --year=2018 --month=1
# python create_text_output.py --model="window_5_years_quarterly" --year=2018 --month=4
# python create_text_output.py --model="window_5_years_quarterly" --year=2018 --month=7
# python create_text_output.py --model="window_5_years_quarterly" --year=2018 --month=10
# python create_text_output.py --model="window_5_years_quarterly" --year=2017 --month=1
# python create_text_output.py --model="window_5_years_quarterly" --year=2017 --month=4
# python create_text_output.py --model="window_5_years_quarterly" --year=2017 --month=7
# python create_text_output.py --model="window_5_years_quarterly" --year=2017 --month=10

# for y in `seq 2000 2018` 
# do
#     for m in 1 4 7 10
#     do
#         python create_text_output.py --model="window_5_years_quarterly" --year=$y --month=$m
#     done
# done

python create_text_output.py --model="window_5_years_quarterly" --year=2019 --month=1

