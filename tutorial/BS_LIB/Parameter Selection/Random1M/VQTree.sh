#！/bin/bash
#This is for running VQTree index

datasize=1000

if [ $datasize == 10000]
then
    

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
for i in {$nc1_start..$nc1_end..$nc1_step}
do
    for j in {$nc2_start..$nc2_end..$nc2_step}
    do
        echo "Running VQTree index with parameter setting: $i $j" 
        ./VQTree $i $j $record_count
        let record_count=$record_count+1
    done
done

echo "Finished VQTree experiments"

