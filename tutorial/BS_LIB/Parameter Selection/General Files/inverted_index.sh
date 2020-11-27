#！/bin/bash
#This is for running inverted index

datasize=10

nc_start=0
nc_end=0
nc_step=0

if [ $datasize == 10000  ]
then
    nc_start=2000
    nc_end=50000
    nc_step=1000
elif [ $datasize == 9000  ]
then
    nc_start=1800
    nc_end=45000
    nc_step=900
elif [ $datasize == 8000  ]
then
    nc_start=1600
    nc_end=40000
    nc_step=800
elif [ $datasize == 7000  ]
then
    nc_start=1400
    nc_end=35000
    nc_step=700
elif [ $datasize == 6000  ]
then
    nc_start=1200
    nc_end=30000
    nc_step=600
elif [ $datasize == 5000 ]
then 
    nc_start=1000
    nc_end=25000
    nc_step=500
elif [ $datasize == 4000 ]
then 
    nc_start=800
    nc_end=20000
    nc_step=400
elif [ $datasize == 3000 ]
then 
    nc_start=600
    nc_end=150000
    nc_step=300
elif [ $datasize == 2000 ]
then 
    nc_start=400
    nc_end=10000
    nc_step=200
elif [ $datasize == 1000 ]
then 
    #nc_start=200
    #nc_end=5000
    #nc_step=200
    nc_start=100
    nc_end=1100
    nc_step=100
elif [ $datasize == 100 ]
then 
    #nc_start=100
    #nc_end=3000
    #nc_step=50
    nc_start=100
    nc_end=1000
    nc_step=100
elif [ $datasize == 10 ]
then 
    #nc_start=50
    #nc_end=1000
    #nc_step=10
    nc_start=10
    nc_end=11
    nc_step=1
fi

record_count=0
time=$(date "+%Y%m%d-%H%M%S")
for ((i=$nc_start; i<=nc_end; i=$i+$nc_step))
do
    echo "Running II index with parameter setting: " $i 
    if [ $datasize == 10000  ]
    then
        ./inverted_index_10000 $i $record_count $time
    elif [ $datasize == 9000  ]
    then
        ./inverted_index_9000 $i $record_count $time
    elif [ $datasize == 8000  ]
    then
        ./inverted_index_8000 $i $record_count $time
    elif [ $datasize == 7000  ]
    then
        ./inverted_index_7000 $i $record_count $time
    elif [ $datasize == 6000  ]
    then
        ./inverted_index_6000 $i $record_count $time
    elif [ $datasize == 5000 ]
    then 
        ./inverted_index_5000 $i $record_count $time
    elif [ $datasize == 4000 ]
    then 
        ./inverted_index_4000 $i $record_count $time
    elif [ $datasize == 3000 ]
    then 
        ./inverted_index_3000 $i $record_count $time
    elif [ $datasize == 2000 ]
    then 
        ./inverted_index_2000 $i $record_count $time
    elif [ $datasize == 1000 ]
    then 
        ./inverted_index_1000 $i $record_count $time
    elif [ $datasize == 100 ]
    then 
        ./inverted_index_100 $i $record_count $time
    elif [ $datasize == 10 ]
    then 
        ./inverted_index_10 $i $record_count $time
    fi
    let record_count=$record_count+1
done


echo "Finished inverted index experiments"




