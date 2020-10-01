#！/bin/bash
#This is for running inverted index from 200 to 5000
count=0
for i in {50..200..20}
do
    for j in {50..200..20}
    do
        ./VQTree $i $j $count
        count++
    done
done

echo "Finished VQTree experiments"
