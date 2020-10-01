#！/bin/bash
#This is for running inverted index from 200 to 5000
count=0
for i in {200..5000..100}
#for i in {1200..5000..100}
do
    ./inverted_index $i $count
    let count=$count+1
done

echo "Finished inverted index experiments"




