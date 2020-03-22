
# Tuning one model (2018 Q1) using five-year rolling window (2013 Q1 to 2017 Q4)
for j in `seq 1 5` 
do
    for i in `seq 1 4` 
    do
        python pci.py --model="window_5_years_quarterly" --year=2018 --month=1 --gpu=0 --iterator=$i
    done
done
