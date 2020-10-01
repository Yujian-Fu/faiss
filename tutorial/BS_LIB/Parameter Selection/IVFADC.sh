#！/bin/bash
#This is for running IVFADC

for first_centroid in {100..2000..100}
do
    min_second_centroid = first_centroid / 50
    max_second_centroid = first_centroid / 10

    for second_centroid in {min_second_centroid..max_second_centroid..min_second_centroid}
    do
        ./IVFADC $first_centroid $second_centroid
    done
done

echo "Finished IVFADC experiments"
