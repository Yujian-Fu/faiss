#!/bin/bash
#---------------
#Run all other files
#---------------
echo 'Running all scripts for SIFT1M dataset'

./inverted_index 1800 0
./inverted_index 2000 1

./ICI 60 5 0
./ICI 48 4 1

./IMI 7 0
./IMI 8 1

./IVFADC 1300 26 0
./IVFADC 700 70 1

./VQTree 40 50 0
./VQTree 50 50 1
