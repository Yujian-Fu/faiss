#！/bin/bash
#This is for running inverted index from 200 to 5000
count=0
for i in {4..128..4}
#for i in {4..32..2}
do
    for j in {3..6..1}
    do
        ./ICI $i $j $count
        let count=$count+1
    done
done

echo "Finished ICI experiments"
