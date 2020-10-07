#!/bin/bash
#---------------
#Run all other files
#---------------
echo 'Running all scripts for DEEP1M dataset'

./inverted_index 500 0
./inverted_index 700 1

./ICI 44 5 0
./ICI 52 5 1

./IMI 7 0
./IMI 8 1

./IVFADC 700 70 0

./VQTree 40 40 0
