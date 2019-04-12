for j in `seq 1 200`
do
    for i in `seq 1 4` 
    do
        python pci.py --model="window_10_years_quarterly" --year=2019 --month=1 --gpu=0 --iterator=$i
    done
done
