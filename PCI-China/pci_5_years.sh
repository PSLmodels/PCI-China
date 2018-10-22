for j in `seq 1 5` 
do
    for y in `seq 1951 2018` 
    do
        for q in `seq 1 4` 
        do
            for i in `seq 1 4` 
            do
                python pci.py --model="window_5_years" --year=$y --quarter=$q --gpu=0 --iterator=$i
            done
        done
    done
done

