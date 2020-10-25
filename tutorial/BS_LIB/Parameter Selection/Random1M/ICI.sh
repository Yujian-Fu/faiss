#！/bin/bash
# This is for running ICI for datasets

datasize=1000

if [ $datasize == 10000 ]
then
    nc_start=0
    nc_end=0
    nc_step=0

    nbits_start=0
    nbits_end=0
    nbits_step=0
elif [ $datasize == 1000 ]
then
    nc_start=4
    nc_end=128
    nc_step=4

    nbits_start=3
    nbits_end=5
    nbits_step=1
elif [ $datasize == 100]
then 
    nc_start=0
    nc_end=0
    nc_step=0

    nbits_start=0
    nbits_end=0
    nbits_step=0
elif [ $datasize == 10]
then
    nc_start=0
    nc_end=0
    nc_step=0

    nbits_start=0
    nbits_end=0
    nbits_step=0
fi

record_count=0
for ((i=$nc_start; i<=$nc_end; i=$i+$nc_step))
do
    for ((j=$nbits_start; j<=$nbits_end; j=$j+$nbits_step))
    do
        echo "Running ICI index with parameter setting: $i $j" 
        ./ICI $i $j $count
        let record_count=$record_count+1
    done
done

echo "Finished ICI experiments"
