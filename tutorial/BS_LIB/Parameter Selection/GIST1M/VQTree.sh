#！/bin/bash
#This is for running inverted index from 200 to 5000
count=0

#for i in {10..50..5}
for i in {10..30..5}
do
    #for j in {10..50..5}
    for j in {10..30..5}
    do
        ./VQTree $i $j $count
        let count=$count+1
    done
done

echo "Finished VQTree experiments"

