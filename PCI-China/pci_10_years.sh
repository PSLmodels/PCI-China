for j in `seq 1 5` 
do
    for y in `seq 1956 2018` 
    do
        for m in `seq 1 12` 
        do
            for i in `seq 1 4` 
            do
                python pci.py --model="window_10_years" --year=$y --month=$m --gpu=0 --iterator=$i
            done
        done
    done
done

