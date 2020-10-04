#！/bin/bash
#This is for running IVFADC
count=0
#for first_centroid in {100..2000..100}

for first_centroid in {100..500..10}
do
    let min_second_centroid=first_centroid/50
    let max_second_centroid=first_centroid/10
    let second_centroid_step=first_centroid/25
    

    for ((second_centroid=$min_second_centroid; second_centroid<=$max_second_centroid; second_centroid+=$second_centroid_step))
    do
        ./IVFADC $first_centroid $second_centroid $count
        echo first_centroid $first_centroid,second_centroid $second_centroid
        let count=$count+1
    done
done

echo "Finished IVFADC experiments"
