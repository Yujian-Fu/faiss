#！/bin/bash
#This is for running inverted index from 200 to 5000
count=0
for i in {200..3000..100}
#for i in {50..1000..50}
do
    ./inverted_index $i $count
    let count=$count+1
done

echo "Finished inverted index experiments"




