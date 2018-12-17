for j in `seq 1 1000` 
do
    for y in `seq 1991 1991` 
    do
        for q in 1
        do
            for i in `seq 1 8` 
            do
                python pci.py --model="window_5_years_quarterly" --year=$y --month=$q --gpu=0 --iterator=$i
            done
        done
    done
done
# python pci.py --model="window_5_years" --year=2010 --month=1 --gpu=0 --iterator=1
