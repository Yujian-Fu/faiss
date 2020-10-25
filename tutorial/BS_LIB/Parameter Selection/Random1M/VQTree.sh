#！/bin/bash
#This is for running VQTree index

datasize=1000

if [ $datasize == 10000]
then
    nc1_start=0
    nc1_end=0
    nc1_step=0

    nc2_start=0
    nc2_end=0
    nc2_step=0
elif [ $datasize == 1000]
then
    nc1_start=10
    nc1_end=50
    nc1_step=10

    nc2_start=10
    nc2_end=50
    nc2_step=10
elif [ $datasize == 100]
then 
    nc1_start=0
    nc1_end=0
    nc1_step=0

    nc2_start=0
    nc2_end=0
    nc2_step=0
elif [ $datasize == 10]
then
    nc1_start=10
    nc1_end=30
    nc1_step=5

    nc2_start=10
    nc2_end=30
    nc2_step=5
fi

record_count=0
for ((i=$nc1_start; i<=$nc1_end; i=$i+$nc1_step))
do
    for ((j=$nc2_start; j<=$nc2_end; j=$j+$nc2_step))
    do
        echo "Running VQTree index with parameter setting: $i $j" 
        ./VQTree $i $j $record_count
        let record_count=$record_count+1
    done
done

echo "Finished VQTree experiments"

