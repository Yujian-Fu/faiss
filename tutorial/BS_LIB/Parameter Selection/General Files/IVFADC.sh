#！/bin/bash
#This is for running IVFADC index


datasize=1000

if [ $datasize == 10000]
then
    nc_start=0
    nc_end=0
    nc_step=0
elif [ $datasize == 1000 ]
then
    nc_start=100
    nc_end=2000
    nc_step=200
elif [ $datasize == 100 ]
then 
    nc_start=0
    nc_end=0
    nc_step=0
elif [ $datasize == 10 ]
then
    nc_start=100
    nc_end=500
    nc_step=10
fi


count=0
for ((first_centroid=$nc_start; first_centroid<=$nc_end; first_centroid=$first_centroid+$nc_step))
do
    let min_second_centroid=first_centroid/50
    let max_second_centroid=first_centroid/10
    let second_centroid_step=first_centroid/25
    

    for ((second_centroid=$min_second_centroid; second_centroid<=$max_second_centroid; second_centroid+=$second_centroid_step))
    do
        echo "Running IVFADC index with parameter setting: $first_centroid $second_centroid" 
        ./IVFADC $first_centroid $second_centroid $count
        echo first_centroid $first_centroid,second_centroid $second_centroid
        let count=$count+1
    done
done

echo "Finished IVFADC experiments"
