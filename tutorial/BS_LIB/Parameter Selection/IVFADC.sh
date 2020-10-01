#！/bin/bash
#This is for running IVFADC
count=0
for first_centroid in {100..2000..100}
do
    let min_second_centroid=first_centroid/50
    let max_second_centroid=first_centroid/10
    

    for ((second_centroid=$min_second_centroid; second_centroid<=$max_second_centroid; second_centroid+=$min_second_centroid))
    do
        ./IVFADC $first_centroid $second_centroid $count
        let count=$count+1
    done
done

echo "Finished IVFADC experiments"
