#！/bin/bash
#This is for running inverted index from 200 to 5000

for i in {50..200..20}
do
    for j in {50..200..20}
    do
        ./inverted_index $i $j
    done
done

echo "Finished VQTree experiments"
