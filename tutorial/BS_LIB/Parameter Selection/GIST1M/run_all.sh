#!/bin/bash
#---------------
#Run all other files
#---------------
echo 'Running all scripts for GIST1M dataset'

./inverted_index 600 0
./inverted_index 2000 1

./ICI 68 4 0
./ICI 68 5 1

./IMI 7 0

./IVFADC 1300 26 0
./IVFADC 900 54 1

./VQTree 30 50 0
