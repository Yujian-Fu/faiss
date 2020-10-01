#！/bin/bash
#This is for running inverted index from 200 to 5000
count=0
for i in {4..8..1}
do
    
    ./IMI $i $count
    let count=$count+1
done

echo "Finished IMI experiments"


