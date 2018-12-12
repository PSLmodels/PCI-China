for y in `seq 2010 2018` 
do
    for q in 1 4 7 10
    do
        for i in `seq 1 4` 
        do
            python pci.py --model="window_5_years_quarterly" --year=$y --month=$q --gpu=0 --iterator=$i
        done
    done
done
# python pci.py --model="window_5_years" --year=2010 --month=1 --gpu=0 --iterator=1
