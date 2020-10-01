#！/bin/bash
#This is for running inverted index from 200 to 5000

for i in {200..5000..100}
do
    ./inverted_index $i 
done

echo "Finished inverted index experiments"




