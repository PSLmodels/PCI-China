# Process data
python proc_data.py 

# Train models and construct PCI (2 years rolling window) 
for j in `seq 1 5` 
do
    for y in `seq 1951 2019` 
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

# Train models and construct PCI (5 years rolling window) 
for j in `seq 1 5` 
do
    for y in `seq 1951 2019` 
    do
        for q in 1 4 7 10
        do
            for i in `seq 1 4` 
            do
                python pci.py --model="window_5_years_quarterly" --year=$y --month=$q --gpu=0 --iterator=$i
            done
        done
    done
done

# Train models and construct PCI (10 years rolling window) 
for j in `seq 1 5` 
do
    for y in `seq 1951 2019` 
    do
        for q in 1 4 7 10
        do
            for i in `seq 1 4` 
            do
                python pci.py --model="window_10_years_quarterly" --year=$y --month=$q --gpu=0 --iterator=$i
            done
        done
    done
done

## Compile all the results together
python compile_tuning.py
python create_text_output.py

## Generate figures 
Rscript gen_figures.R --vanilla --verbose

## Generate plotly figure 
python create_plotly.py
