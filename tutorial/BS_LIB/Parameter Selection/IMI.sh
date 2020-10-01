#！/bin/bash
#This is for running inverted index from 200 to 5000

for i in {4..8..1}
do
    count=0
    ./IMI $i $count
    let count=$count+1
done

echo "Finished IMI experiments"


