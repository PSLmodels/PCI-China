for y in `seq 1951 2018` 
do
    for q in 1 4 7 10
    do
        python pci.py --model="window_5_years" --year=$y --month=$q --gpu=0 --iterator=1
    done
done
# python pci.py --model="window_5_years" --year=2010 --month=1 --gpu=0 --iterator=1
