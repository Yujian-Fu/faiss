#！/bin/bash
#This is for running inverted index from 200 to 5000

for i in {200..5000..100}
do
    count=0
    ./inverted_index $i $count
    count+=1
done

echo "Finished inverted index experiments"




