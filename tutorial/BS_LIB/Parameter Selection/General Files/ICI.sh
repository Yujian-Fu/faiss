#！/bin/bash
# This is for running ICI for datasets

datasize=10

if [ $datasize == 10000 ]
then
    nc_start=50
    nc_end=500
    nc_step=10

    nbits_start=3
    nbits_end=6
    nbits_step=1
elif [ $datasize == 1000 ]
then
    nc_start=10
    nc_end=100
    nc_step=5

    nbits_start=3
    nbits_end=5
    nbits_step=1
elif [ $datasize == 100]
then 
    nc_start=5
    nc_end=50
    nc_step=5

    nbits_start=3
    nbits_end=5
    nbits_step=1
elif [ $datasize == 10]
then
    nc_start=5
    nc_end=20
    nc_step=2

    nbits_start=3
    nbits_end=4
    nbits_step=1
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
