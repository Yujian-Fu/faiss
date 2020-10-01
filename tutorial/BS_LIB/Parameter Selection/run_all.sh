#!/bin/bash
#---------------
#Run all other files
#---------------
echo 'Running all scripts for SIFT1M dataset with 8 bytes'

echo 'Running Inverted Index'
source inverted_index.sh

echo 'Running IMI'
source IMI.sh

echo 'Running ICI'
source ICI.sh

echo 'VQTree'
source VQTree.sh

echo 'Running IVFADC'
source IVFADC.sh