#！/bin/bash
#This is for running IMI index


datasize=1000

if [ $datasize == 10000  ]
then
    nbits_start=0
    nbits_end=0
    nbits_step=0
elif [ $datasize == 1000 ]
then 
    nbits_start=4
    nbits_end=8
    nbits_step=1
elif [ $datasize == 100 ]
then 
    nbits_start=0
    nbits_end=0
    nbits_step=0
elif [ $datasize == 10 ]
then 
    nbits_start=3
    nbits_end=7
    nbits_step=1

fi


record_count=0
for ((i=$nbits_start; i<=$nbits_end; i=$i+$nbits_step}
do
    echo "Running IMI index with parameter setting: $i" 
    ./IMI $i $record_count
    let record_count=$record_count+1
done

echo "Finished IMI experiments"



