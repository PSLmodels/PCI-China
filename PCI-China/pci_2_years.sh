for j in `seq 1 5` 
do
    for y in `seq 1950 2018` 
    do
        for q in 1 4 7 10
        do
            for i in `seq 1 4` 
            do
                python pci.py --model="window_2_years_quarterly" --year=$y --month=$q --gpu=0 --iterator=$i
            done
        done
    done
done
