#！/bin/bash
#This is for running inverted index

datasize=1000

if [ $datasize == 10000  ]
then
    nc_start=0
    nc_end=0
    nc_step=0
elif [ $datasize == 1000 ]
then 
    nc_start=200
    nc_end=3000
    nc_step=100
elif [ $datasize == 100 ]
then 
    nc_start=0
    nc_end=0
    nc_step=0
elif [ $datasize == 10 ]
then 
    nc_start=50
    nc_end=1000
    nc_step=50

fi


record_count=0
for ((i=$nc_start; i<=nc_end; i=$i+$nc_step))
do
    echo "Running II index with parameter setting: " $i 
    ./inverted_index $i $record_count
    let record_count=$record_count+1
done

echo "Finished inverted index experiments"




