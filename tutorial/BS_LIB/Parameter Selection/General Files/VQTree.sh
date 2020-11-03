#！/bin/bash
#This is for running VQTree index

datasize=1000

nc1_start=0
nc1_end=0
nc1_step=0
nc2_start=0
nc2_end=0
nc2_step=0

if [ $datasize == 10000 ]
then
    nc1_start=50
    nc1_end=300
    nc1_step=50
    nc2_start=50
    nc2_end=300
    nc2_step=50
elif [ $datasize == 1000 ]
then
    nc1_start=10
    nc1_end=100
    nc1_step=5
    nc2_start=10
    nc2_end=100
    nc2_step=5
elif [ $datasize == 100 ]
then 
    nc1_start=10
    nc1_end=100
    nc1_step=10

    nc2_start=10
    nc2_end=100
    nc2_step=10
elif [ $datasize == 10 ]
then
    nc1_start=10
    nc1_end=30
    nc1_step=5
    nc2_start=10
    nc2_end=30
    nc2_step=5
fi


record_count=0
time=$(date "+%Y%m%d-%H%M%S")
for ((i=$nc1_start; i<=$nc1_end; i=$i+$nc1_step))
do
    for ((j=$nc2_start; j<=$nc2_end; j=$j+$nc2_step))
    do
        echo "Running VQTree index with parameter setting: $i $j" 
        if [ $datasize == 10000 ]
        then
            ./VQTree_10000 $i $j $record_count $time
        elif [ $datasize == 1000 ]
        then
            ./VQTree_1000 $i $j $record_count $time
        elif [ $datasize == 100 ]
        then 
            ./VQTree_100 $i $j $record_count $time
        elif [ $datasize == 10 ]
        then
            ./VQTree_10 $i $j $record_count $time
        fi

        let record_count=$record_count+1
    done
done

echo "Finished VQTree experiments"

