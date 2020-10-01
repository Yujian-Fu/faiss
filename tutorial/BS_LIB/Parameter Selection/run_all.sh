#!/bin/bash
#---------------
#Run all other files
#---------------
echo 'Running all scripts for DEEP1M dataset with 16bytes'

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