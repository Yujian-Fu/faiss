#！/bin/bash
#This is for running inverted index from 200 to 5000

for i in {4..64..2}
do
    for j in {4..8..1}
    do
        ./inverted_index $i $j
    done
done

echo "Finished ICI experiments"
